package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

const (
	wsMsgTypeStartupConfig = "startup_config"
	wsMsgTypeSignal        = "signal"
	wsMsgTypeSignalReady   = "signal_ready"
	wsMsgTypeError         = "error"

	wsPingInterval = 30 * time.Second
	wsPingTimeout  = 10 * time.Second
)

// wsSession is an active WebSocket connection for a registered node.
type wsSession struct {
	nodeID string
	conn   *websocket.Conn
	mu     sync.Mutex // serialises concurrent writes
}

func (sess *wsSession) writeJSON(ctx context.Context, v any) error {
	sess.mu.Lock()
	defer sess.mu.Unlock()
	return wsjson.Write(ctx, sess.conn, v)
}

// SetSignalingOptions configures the WebSocket auth token and signaling deadline.
// Call this after NewServer and before serving requests.
func (s *Server) SetSignalingOptions(clusterToken string, signalTimeout time.Duration) {
	s.clusterToken = clusterToken
	if signalTimeout > 0 {
		s.signalTimeout = signalTimeout
	}
}

func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	nodeID := r.URL.Query().Get("node_id")
	token := r.URL.Query().Get("token")

	if nodeID == "" {
		http.Error(w, "missing node_id", http.StatusBadRequest)
		return
	}
	if s.clusterToken != "" && token != s.clusterToken {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	s.mu.RLock()
	_, known := s.nodes[nodeID]
	s.mu.RUnlock()
	if !known {
		http.Error(w, "unknown node_id", http.StatusForbidden)
		return
	}

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify: true,
	})
	if err != nil {
		log.Printf("[ws] upgrade failed node=%s: %v", nodeID, err)
		return
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	sess := &wsSession{nodeID: nodeID, conn: conn}

	s.wsSessionsMu.Lock()
	s.wsSessions[nodeID] = sess
	s.wsSessionsMu.Unlock()

	defer func() {
		s.wsSessionsMu.Lock()
		if s.wsSessions[nodeID] == sess {
			delete(s.wsSessions, nodeID)
		}
		s.wsSessionsMu.Unlock()
		log.Printf("[ws] session closed: node=%s", nodeID)
	}()

	log.Printf("[ws] session established: node=%s", nodeID)

	// Push startup_config immediately if the cluster is already configured.
	if wsMsg, ok := s.buildStartupConfigWSMsg(nodeID); ok {
		pushCtx, pushCancel := context.WithTimeout(ctx, 10*time.Second)
		if err := sess.writeJSON(pushCtx, wsMsg); err != nil {
			pushCancel()
			log.Printf("[ws] startup_config on-connect push failed node=%s: %v", nodeID, err)
			return
		}
		pushCancel()
		log.Printf("[ws] startup_config pushed on connect: node=%s", nodeID)
	}

	go s.wsPingLoop(ctx, sess)

	for {
		var raw map[string]json.RawMessage
		if err := wsjson.Read(ctx, conn, &raw); err != nil {
			if ctx.Err() != nil {
				return
			}
			return
		}
		switch wsStringField(raw, "type") {
		case wsMsgTypeSignal:
			data, _ := json.Marshal(raw)
			var sig SignalMsg
			if err := json.Unmarshal(data, &sig); err == nil {
				s.receiveNodeSignal(nodeID, sig)
			}
		}
	}
}

// buildStartupConfigWSMsg returns the per-node startup_config wrapped with a
// "type" field, ready to send over the wire. Returns false when not yet ready.
func (s *Server) buildStartupConfigWSMsg(nodeID string) (map[string]any, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.startupConfig == nil {
		return nil, false
	}
	cfg := *s.startupConfig
	if a, ok := s.assignments[nodeID]; ok {
		ac := a
		cfg.Assignment = &ac
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		return nil, false
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, false
	}
	m["type"] = wsMsgTypeStartupConfig
	return m, true
}

// pushStartupConfigOverWS pushes startup configs to nodes that already have a
// WS session. Returns targets for which no WS session exists (HTTP fallback).
func (s *Server) pushStartupConfigOverWS(targets []StartupTarget) []StartupTarget {
	var httpFallback []StartupTarget

	s.wsSessionsMu.RLock()
	for _, target := range targets {
		sess, ok := s.wsSessions[target.Node.NodeID]
		if !ok {
			httpFallback = append(httpFallback, target)
			continue
		}
		go func(sess *wsSession, target StartupTarget) {
			b, err := json.Marshal(target.Config)
			if err != nil {
				return
			}
			var m map[string]any
			if err := json.Unmarshal(b, &m); err != nil {
				return
			}
			m["type"] = wsMsgTypeStartupConfig

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := sess.writeJSON(ctx, m); err != nil {
				log.Printf("[ws] startup_config push failed node=%s: %v", target.Node.NodeID, err)
				return
			}
			log.Printf("[ws] startup_config delivered: node=%s cluster=%s stage=%d/%d",
				target.Node.NodeID, target.Config.ClusterID,
				target.Config.Assignment.StageIndex, target.Config.Assignment.StageCount)
		}(sess, target)
	}
	s.wsSessionsMu.RUnlock()

	return httpFallback
}

func (s *Server) wsPingLoop(ctx context.Context, sess *wsSession) {
	ticker := time.NewTicker(wsPingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pCtx, cancel := context.WithTimeout(ctx, wsPingTimeout)
			err := sess.conn.Ping(pCtx)
			cancel()
			if err != nil {
				return
			}
		}
	}
}

func (s *Server) receiveNodeSignal(nodeID string, sig SignalMsg) {
	ns := NodeSignal{
		NodeID:       nodeID,
		ExternalAddr: sig.ExternalAddr,
		ExternalPort: sig.ExternalPort,
		TransportMode: sig.TransportMode,
	}
	log.Printf("[ws] signal received: node=%s addr=%s:%d mode=%s",
		nodeID, sig.ExternalAddr, sig.ExternalPort, sig.TransportMode)

	s.mu.Lock()
	isFirst := len(s.nodeSignals) == 0
	s.nodeSignals[nodeID] = ns
	allSignaled := s.startupConfig != nil && len(s.nodeSignals) >= len(s.nodeOrder)
	nodeOrder := append([]string(nil), s.nodeOrder...)
	clusterID := s.clusterID
	peers := make([]NodeSignal, 0, len(s.nodeSignals))
	for _, n := range s.nodeSignals {
		peers = append(peers, n)
	}
	s.mu.Unlock()

	if isFirst {
		s.startSignalingDeadline(clusterID, nodeOrder)
	}
	if allSignaled {
		s.cancelSignalingDeadline()
		s.broadcastSignalReady(clusterID, peers)
	}
}

func (s *Server) startSignalingDeadline(clusterID string, nodeOrder []string) {
	s.signalingTimerOnce.Do(func() {
		timer := time.AfterFunc(s.signalTimeout, func() {
			s.mu.RLock()
			signaled := make(map[string]bool, len(s.nodeSignals))
			for k := range s.nodeSignals {
				signaled[k] = true
			}
			s.mu.RUnlock()

			var missing []string
			for _, id := range nodeOrder {
				if !signaled[id] {
					missing = append(missing, id)
				}
			}
			if len(missing) == 0 {
				return
			}
			log.Printf("[ws] signal_timeout: cluster=%s missing=%v", clusterID, missing)

			errMsg := ErrorMsg{
				Type:    wsMsgTypeError,
				Code:    "signal_timeout",
				Message: fmt.Sprintf("nodes %v never signaled", missing),
			}
			s.wsSessionsMu.RLock()
			sessions := make([]*wsSession, 0, len(s.wsSessions))
			for _, sess := range s.wsSessions {
				sessions = append(sessions, sess)
			}
			s.wsSessionsMu.RUnlock()

			for _, sess := range sessions {
				go func(sess *wsSession) {
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()
					_ = sess.writeJSON(ctx, errMsg)
					_ = sess.conn.Close(websocket.StatusNormalClosure, "signal timeout")
				}(sess)
			}
		})
		s.signalingTimerMu.Lock()
		s.signalingTimer = timer
		s.signalingTimerMu.Unlock()
	})
}

func (s *Server) cancelSignalingDeadline() {
	s.signalingTimerMu.Lock()
	if s.signalingTimer != nil {
		s.signalingTimer.Stop()
	}
	s.signalingTimerMu.Unlock()
}

func (s *Server) broadcastSignalReady(clusterID string, peers []NodeSignal) {
	msg := SignalReadyMsg{
		Type:      wsMsgTypeSignalReady,
		ClusterID: clusterID,
		Peers:     peers,
	}
	s.wsSessionsMu.RLock()
	sessions := make([]*wsSession, 0, len(s.wsSessions))
	for _, sess := range s.wsSessions {
		sessions = append(sessions, sess)
	}
	s.wsSessionsMu.RUnlock()

	log.Printf("[ws] broadcasting signal_ready: cluster=%s peers=%d sessions=%d",
		clusterID, len(peers), len(sessions))

	for _, sess := range sessions {
		go func(sess *wsSession, msg SignalReadyMsg) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := sess.writeJSON(ctx, msg); err != nil {
				log.Printf("[ws] signal_ready push failed node=%s: %v", sess.nodeID, err)
			}
		}(sess, msg)
	}
}

func wsStringField(m map[string]json.RawMessage, key string) string {
	raw, ok := m[key]
	if !ok {
		return ""
	}
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}
