package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Server struct {
	minNodes       int
	modelName      string
	rayHeadAddress string
	executionMode  string
	backendConfig  BackendConfig
	client         *http.Client

	mu            sync.RWMutex
	nodes         map[string]NodeInfo
	nodeRuntime   map[string]NodeRuntimeStatus
	nodeOrder     []string
	clusterID     string
	assignments   map[string]NodeAssignment
	startupConfig *StartupConfig

	// WebSocket signaling
	clusterToken      string
	signalTimeout     time.Duration
	wsSessionsMu      sync.RWMutex
	wsSessions        map[string]*wsSession
	nodeSignals       map[string]NodeSignal
	signalingTimer    *time.Timer
	signalingTimerMu  sync.Mutex
	signalingTimerOnce sync.Once
}

func NewServer(minNodes int, modelName, rayHeadAddress, executionMode string, backendConfig BackendConfig) *Server {
	return &Server{
		minNodes:       minNodes,
		modelName:      modelName,
		rayHeadAddress: rayHeadAddress,
		executionMode:  executionMode,
		backendConfig:  cloneBackendConfig(backendConfig),
		client:         &http.Client{},
		nodes:          make(map[string]NodeInfo),
		nodeRuntime:    make(map[string]NodeRuntimeStatus),
		assignments:    make(map[string]NodeAssignment),
		wsSessions:     make(map[string]*wsSession),
		nodeSignals:    make(map[string]NodeSignal),
		signalTimeout:  60 * time.Second,
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/config", s.handleConfig)
	mux.HandleFunc("/register", s.handleRegister)
	mux.HandleFunc("/node-status", s.handleNodeStatus)
	mux.HandleFunc("/ws", s.handleWS)
	mux.HandleFunc("/v1/", s.handleV1Proxy)
	return mux
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	status := s.BuildStatus()
	writeJSON(w, http.StatusOK, status)
}

func (s *Server) handleConfig(w http.ResponseWriter, _ *http.Request) {
	s.mu.RLock()
	cfg := s.startupConfig
	s.mu.RUnlock()
	if cfg == nil {
		writeError(w, http.StatusNotFound, "Pipeline has not started yet.")
		return
	}
	writeJSON(w, http.StatusOK, cfg)
}

func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var registration NodeRegistration
	if err := json.NewDecoder(r.Body).Decode(&registration); err != nil {
		writeError(w, http.StatusBadRequest, "invalid registration payload")
		return
	}
	if registration.NodeID == "" || registration.Host == "" || registration.Port <= 0 {
		writeError(w, http.StatusBadRequest, "missing required registration fields")
		return
	}
	if registration.CallbackURL == "" {
		registration.CallbackURL = fmt.Sprintf("http://%s:%d", registration.Host, registration.Port)
	}
	if registration.WorkerURL == "" {
		registration.WorkerURL = fmt.Sprintf("http://%s:8100", registration.Host)
	}

	var (
		response RegistrationResponse
		cfg      *StartupConfig
		targets  []StartupTarget
	)

	s.mu.Lock()
	if s.startupConfig != nil {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, "Cluster startup already triggered; dynamic resize is disabled.")
		return
	}

	nodeInfo := NodeInfo{
		NodeID:      registration.NodeID,
		Host:        registration.Host,
		Port:        registration.Port,
		VRAMGB:      registration.VRAMGB,
		CallbackURL: strings.TrimRight(registration.CallbackURL, "/"),
		WorkerURL:   strings.TrimRight(registration.WorkerURL, "/"),
	}
	if _, exists := s.nodes[registration.NodeID]; !exists {
		s.nodeOrder = append(s.nodeOrder, registration.NodeID)
	}
	s.nodes[registration.NodeID] = nodeInfo
	s.nodeRuntime[registration.NodeID] = NodeRuntimeStatus{
		NodeID:          registration.NodeID,
		LifecycleState:  "registered",
		LifecycleDetail: "Registration accepted by coordinator.",
		WorkerURL:       nodeInfo.WorkerURL,
		UpdatedAt:       time.Now().UTC(),
	}

	response = RegistrationResponse{
		Accepted:        true,
		RegisteredNodes: len(s.nodes),
	}

	if len(s.nodeOrder) >= s.minNodes && s.startupConfig == nil {
		clusterCfg, startupTargets := s.BuildClusterStartupConfigLocked()
		cfg = &clusterCfg
		targets = startupTargets
		s.startupConfig = cfg
		response.StartupTriggered = true
	}
	s.mu.Unlock()

	if cfg != nil {
		go s.broadcastStartup(targets)
	}

	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleNodeStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var update NodeStatusUpdate
	if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
		writeError(w, http.StatusBadRequest, "invalid node status payload")
		return
	}
	if update.NodeID == "" || update.LifecycleState == "" {
		writeError(w, http.StatusBadRequest, "missing required node status fields")
		return
	}

	s.mu.Lock()
	if _, ok := s.nodes[update.NodeID]; !ok {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, "unknown node_id")
		return
	}
	s.applyNodeStatusLocked(update)
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]bool{"accepted": true})
}

func (s *Server) handleV1Proxy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost && r.Method != http.MethodPut && r.Method != http.MethodDelete {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	status := s.BuildStatus()
	if !status.InferenceReady {
		writeError(w, http.StatusServiceUnavailable, "Pipeline is not inference-ready")
		return
	}

	node, ok := s.selectedNode()
	if !ok {
		writeError(w, http.StatusServiceUnavailable, "No node available")
		return
	}

	target := strings.TrimRight(node.WorkerURL, "/") + r.URL.Path
	if rawQuery := r.URL.RawQuery; rawQuery != "" {
		target += "?" + rawQuery
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "Failed to read request body")
		return
	}

	req, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to build upstream request")
		return
	}
	copyHeaders(req.Header, r.Header)
	req.Header.Del("Host")
	req.Header.Del("Content-Length")

	resp, err := s.client.Do(req)
	if err != nil {
		log.Printf("proxy to node %s failed: %v", node.NodeID, err)
		writeError(w, http.StatusBadGateway, "Selected node worker unreachable")
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if errors.Is(readErr, io.EOF) {
			return
		}
		if readErr != nil {
			return
		}
	}
}

func (s *Server) BuildStatus() StatusResponse {
	s.mu.RLock()
	nodes := s.snapshotNodeStatusLocked()
	cfg := s.startupConfig
	clusterID := s.clusterID
	modelName := s.modelName
	rayHeadAddress := s.rayHeadAddress
	selected := ""
	executionMode := ""
	stageCount := 0
	entryNodeID := ""
	if cfg != nil {
		selected = cfg.EntryNodeID
		executionMode = cfg.ExecutionMode
		stageCount = cfg.StageCount
		entryNodeID = cfg.EntryNodeID
	}
	if selected == "" && len(s.nodeOrder) > 0 {
		selected = s.nodeOrder[0]
	}
	minNodes := s.minNodes
	s.mu.RUnlock()

	clusterReady := cfg != nil && len(nodes) >= minNodes
	allNodesReady := clusterReady
	backendReady := clusterReady && executionMode != ""
	entryNodeReady := false
	inferenceReady := false

	for _, node := range nodes {
		if clusterReady && !isNodeLifecycleReady(node, executionMode) {
			allNodesReady = false
		}
		if backendReady && !isBackendReadyNode(node, executionMode) {
			backendReady = false
		}
		if node.NodeID != selected {
			continue
		}
		entryLifecycleReady := isNodeLifecycleReady(node, executionMode)
		if isExecutableMode(executionMode) {
			entryNodeReady = entryLifecycleReady && s.checkInferenceReady(node.WorkerURL)
		} else {
			entryNodeReady = entryLifecycleReady
		}
	}

	if !clusterReady {
		allNodesReady = false
		backendReady = false
	}

	pipelineReady := clusterReady && allNodesReady
	if isExecutableMode(executionMode) {
		inferenceReady = pipelineReady && backendReady && entryNodeReady
	}

	return StatusResponse{
		MinNodes:        minNodes,
		RegisteredNodes: len(nodes),
		ClusterReady:    clusterReady,
		EntryNodeReady:  entryNodeReady,
		AllNodesReady:   allNodesReady,
		BackendReady:    backendReady,
		PipelineReady:   pipelineReady,
		InferenceReady:  inferenceReady,
		ClusterID:       clusterID,
		ModelName:       modelName,
		ExecutionMode:   executionMode,
		StageCount:      stageCount,
		RayHeadAddress:  rayHeadAddress,
		EntryNodeID:     entryNodeID,
		SelectedNodeID:  selected,
		Nodes:           nodes,
	}
}

func (s *Server) checkInferenceReady(workerURL string) bool {
	if workerURL == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(workerURL, "/")+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (s *Server) broadcastStartup(targets []StartupTarget) {
	// Deliver over WS to nodes that already have an active session.
	// httpTargets holds nodes without a WS session and need the HTTP fallback.
	httpTargets := s.pushStartupConfigOverWS(targets)

	for _, target := range httpTargets {
		payload, err := json.Marshal(target.Config)
		if err != nil {
			log.Printf("failed to marshal startup payload for node=%s: %v", target.Node.NodeID, err)
			continue
		}

		node := target.Node
		url := strings.TrimRight(node.CallbackURL, "/") + "/startup"
		delivered := false
		for attempt := 1; attempt <= 5; attempt++ {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
			if err != nil {
				cancel()
				break
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := s.client.Do(req)
			cancel()
			if err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
				resp.Body.Close()
				delivered = true
				log.Printf(
					"startup signal delivered to node=%s cluster=%s stage=%d/%d (%s)",
					node.NodeID,
					target.Config.ClusterID,
					target.Config.Assignment.StageIndex,
					target.Config.Assignment.StageCount,
					url,
				)
				break
			}
			if resp != nil {
				resp.Body.Close()
			}
			log.Printf("startup signal failed for node=%s attempt=%d", node.NodeID, attempt)
			time.Sleep(time.Duration(attempt) * 1500 * time.Millisecond)
		}
		if !delivered {
			log.Printf("failed to deliver startup signal to node=%s", node.NodeID)
		}
	}
}

func (s *Server) selectedNode() (NodeInfo, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.startupConfig != nil && s.startupConfig.EntryNodeID != "" {
		node, ok := s.nodes[s.startupConfig.EntryNodeID]
		if ok {
			return node, true
		}
	}
	if len(s.nodeOrder) == 0 {
		return NodeInfo{}, false
	}
	node, ok := s.nodes[s.nodeOrder[0]]
	return node, ok
}

func (s *Server) startupConfigSnapshot() *StartupConfig {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.startupConfig
}

func (s *Server) snapshotNodesLocked() []NodeInfo {
	nodes := make([]NodeInfo, 0, len(s.nodeOrder))
	for _, nodeID := range s.nodeOrder {
		nodes = append(nodes, s.nodes[nodeID])
	}
	return nodes
}

func (s *Server) snapshotNodeStatusLocked() []NodeInfo {
	nodes := make([]NodeInfo, 0, len(s.nodeOrder))
	for _, nodeID := range s.nodeOrder {
		node := s.nodes[nodeID]
		if assignment, ok := s.assignments[nodeID]; ok {
			assignmentCopy := assignment
			stageIndex := assignment.StageIndex
			node.StageIndex = &stageIndex
			node.StageRole = assignment.StageRole
			node.Assignment = &assignmentCopy
		}
		if runtime, ok := s.nodeRuntime[nodeID]; ok {
			node.ClusterID = runtime.ClusterID
			node.LifecycleState = runtime.LifecycleState
			node.LifecycleDetail = runtime.LifecycleDetail
			if runtime.WorkerURL != "" {
				node.WorkerURL = runtime.WorkerURL
			}
			node.UpdatedAt = runtime.UpdatedAt.Format(time.RFC3339)
		}
		nodes = append(nodes, node)
	}
	return nodes
}

func (s *Server) BuildClusterStartupConfigLocked() (StartupConfig, []StartupTarget) {
	s.clusterID = fmt.Sprintf("cluster-%d", time.Now().UTC().UnixNano())
	s.assignments = make(map[string]NodeAssignment, len(s.nodeOrder))

	stageCount := len(s.nodeOrder)
	if stageCount == 0 {
		stageCount = 1
	}
	executionMode := s.executionMode
	if executionMode == "" {
		executionMode = defaultExecutionMode(stageCount)
	}
	entryNodeID := ""
	if len(s.nodeOrder) > 0 {
		entryNodeID = s.nodeOrder[0]
	}

	topology := make([]NodeInfo, 0, len(s.nodeOrder))
	for idx, nodeID := range s.nodeOrder {
		node := s.nodes[nodeID]
		stageIndex := idx
		node.StageIndex = &stageIndex
		node.StageRole = stageRoleForIndex(idx, stageCount)
		topology = append(topology, node)
	}

	backendConfig := cloneBackendConfig(s.backendConfig)
	if backendConfig.RayHeadAddress == "" {
		backendConfig.RayHeadAddress = s.rayHeadAddress
	}

	clusterCfg := StartupConfig{
		ClusterID:            s.clusterID,
		ModelName:            s.modelName,
		ExecutionMode:        executionMode,
		PipelineParallelSize: stageCount,
		StageCount:           stageCount,
		EntryNodeID:          entryNodeID,
		RayHeadAddress:       s.rayHeadAddress,
		Nodes:                topology,
		BackendConfig:        backendConfig,
	}

	targets := make([]StartupTarget, 0, len(topology))
	for idx, topoNode := range topology {
		assignment := NodeAssignment{
			NodeID:       topoNode.NodeID,
			StageIndex:   idx,
			StageCount:   stageCount,
			StageRole:    topoNode.StageRole,
			LoadStrategy: loadStrategyForMode(executionMode),
			SliceSpec: SliceSpec{
				Kind:           "stage_index",
				StageIndex:     idx,
				StageCount:     stageCount,
				PartitionLabel: fmt.Sprintf("%d/%d", idx, stageCount),
				Executable:     executionMode != "dry_run",
			},
			PeerNodes:      buildPeerNodes(topology, idx),
			WorkerEndpoint: topoNode.WorkerURL,
		}
		s.assignments[topoNode.NodeID] = assignment

		nodeCfg := clusterCfg
		assignmentCopy := assignment
		nodeCfg.Assignment = &assignmentCopy
		targets = append(targets, StartupTarget{
			Node:   s.nodes[topoNode.NodeID],
			Config: nodeCfg,
		})
	}

	return clusterCfg, targets
}

func (s *Server) applyNodeStatusLocked(update NodeStatusUpdate) {
	if update.ClusterID == "" && s.clusterID != "" {
		update.ClusterID = s.clusterID
	}
	if update.WorkerURL != "" {
		node := s.nodes[update.NodeID]
		node.WorkerURL = strings.TrimRight(update.WorkerURL, "/")
		s.nodes[update.NodeID] = node
	}
	if update.Assignment != nil {
		s.assignments[update.NodeID] = *update.Assignment
	}
	s.nodeRuntime[update.NodeID] = NodeRuntimeStatus{
		NodeID:          update.NodeID,
		ClusterID:       update.ClusterID,
		LifecycleState:  update.LifecycleState,
		LifecycleDetail: update.LifecycleDetail,
		WorkerURL:       strings.TrimRight(update.WorkerURL, "/"),
		UpdatedAt:       time.Now().UTC(),
	}
}
