package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type NodeRegistration struct {
	NodeID      string  `json:"node_id"`
	Host        string  `json:"host"`
	Port        int     `json:"port"`
	VRAMGB      float64 `json:"vram_gb"`
	CallbackURL string  `json:"callback_url"`
	WorkerURL   string  `json:"worker_url"`
}

type StartupConfig struct {
	ModelName            string `json:"model_name"`
	PipelineParallelSize int    `json:"pipeline_parallel_size"`
	RayHeadAddress       string `json:"ray_head_address"`
}

type RegistrationResponse struct {
	Accepted         bool `json:"accepted"`
	RegisteredNodes  int  `json:"registered_nodes"`
	StartupTriggered bool `json:"startup_triggered"`
}

type NodeInfo struct {
	NodeID      string  `json:"node_id"`
	Host        string  `json:"host"`
	Port        int     `json:"port"`
	VRAMGB      float64 `json:"vram_gb"`
	CallbackURL string  `json:"callback_url"`
	WorkerURL   string  `json:"worker_url"`
}

type StatusResponse struct {
	MinNodes        int        `json:"min_nodes"`
	RegisteredNodes int        `json:"registered_nodes"`
	PipelineReady   bool       `json:"pipeline_ready"`
	InferenceReady  bool       `json:"inference_ready"`
	ModelName       string     `json:"model_name"`
	RayHeadAddress  string     `json:"ray_head_address"`
	SelectedNodeID  string     `json:"selected_node_id,omitempty"`
	Nodes           []NodeInfo `json:"nodes"`
}

type Server struct {
	minNodes       int
	modelName      string
	rayHeadAddress string
	client         *http.Client

	mu            sync.RWMutex
	nodes         map[string]NodeInfo
	nodeOrder     []string
	startupConfig *StartupConfig
}

func NewServer(minNodes int, modelName, rayHeadAddress string) *Server {
	return &Server{
		minNodes:       minNodes,
		modelName:      modelName,
		rayHeadAddress: rayHeadAddress,
		client:         &http.Client{},
		nodes:          make(map[string]NodeInfo),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/config", s.handleConfig)
	mux.HandleFunc("/register", s.handleRegister)
	mux.HandleFunc("/v1/", s.handleV1Proxy)
	return mux
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	status := s.buildStatus()
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
		nodes    []NodeInfo
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

	response = RegistrationResponse{
		Accepted:        true,
		RegisteredNodes: len(s.nodes),
	}

	if len(s.nodeOrder) >= s.minNodes && s.startupConfig == nil {
		cfg = &StartupConfig{
			ModelName:            s.modelName,
			PipelineParallelSize: len(s.nodeOrder),
			RayHeadAddress:       s.rayHeadAddress,
		}
		s.startupConfig = cfg
		nodes = s.snapshotNodesLocked()
		response.StartupTriggered = true
	}
	s.mu.Unlock()

	if cfg != nil {
		go s.broadcastStartup(*cfg, nodes)
	}

	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleV1Proxy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost && r.Method != http.MethodPut && r.Method != http.MethodDelete {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	node, ok := s.selectedNode()
	if !ok {
		writeError(w, http.StatusServiceUnavailable, "No node available")
		return
	}
	if s.startupConfigSnapshot() == nil {
		writeError(w, http.StatusServiceUnavailable, "Pipeline not ready")
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

func (s *Server) buildStatus() StatusResponse {
	s.mu.RLock()
	nodes := s.snapshotNodesLocked()
	cfg := s.startupConfig
	modelName := s.modelName
	rayHeadAddress := s.rayHeadAddress
	selected := ""
	if len(s.nodeOrder) > 0 {
		selected = s.nodeOrder[0]
	}
	minNodes := s.minNodes
	s.mu.RUnlock()

	inferenceReady := false
	if cfg != nil && selected != "" {
		for _, node := range nodes {
			if node.NodeID == selected {
				inferenceReady = s.checkInferenceReady(node.WorkerURL)
				break
			}
		}
	}

	return StatusResponse{
		MinNodes:        minNodes,
		RegisteredNodes: len(nodes),
		PipelineReady:   cfg != nil,
		InferenceReady:  inferenceReady,
		ModelName:       modelName,
		RayHeadAddress:  rayHeadAddress,
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

func (s *Server) broadcastStartup(cfg StartupConfig, nodes []NodeInfo) {
	payload, err := json.Marshal(cfg)
	if err != nil {
		log.Printf("failed to marshal startup payload: %v", err)
		return
	}

	for _, node := range nodes {
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
				log.Printf("startup signal delivered to node=%s (%s)", node.NodeID, url)
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

func copyHeaders(dst, src http.Header) {
	for key, values := range src {
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("failed to encode JSON response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]string{"detail": detail})
}

func detectLocalIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()

	localAddr, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok {
		return "127.0.0.1"
	}
	return localAddr.IP.String()
}

func maybeStartRayHead(enabled bool, port int) {
	if !enabled {
		return
	}

	cmd := exec.Command("ray", "start", "--head", "--port", fmt.Sprintf("%d", port), "--disable-usage-stats")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("ray head start exited with error (this may be expected if already running): %v: %s", err, string(output))
		return
	}
	log.Printf("ray head started successfully")
}

func main() {
	host := flag.String("host", "0.0.0.0", "Coordinator bind host")
	port := flag.Int("port", 8000, "Coordinator bind port")
	minNodes := flag.Int("min-nodes", 2, "Minimum nodes required before startup signal")
	modelName := flag.String("model-name", "", "Model name/path used by all nodes and the user service")
	rayPort := flag.Int("ray-port", 6379, "Ray GCS port for head node")
	rayHeadAddress := flag.String("ray-head-address", "", "Ray head address host:port. If omitted, inferred from local IP + --ray-port")
	autostartRayHead := flag.Bool("autostart-ray-head", false, "Start `ray start --head` before serving HTTP API")
	flag.Parse()

	if *modelName == "" {
		fmt.Fprintln(os.Stderr, "--model-name is required")
		os.Exit(2)
	}
	if *minNodes < 1 {
		fmt.Fprintln(os.Stderr, "--min-nodes must be >= 1")
		os.Exit(2)
	}

	maybeStartRayHead(*autostartRayHead, *rayPort)

	resolvedRayHead := *rayHeadAddress
	if resolvedRayHead == "" {
		resolvedRayHead = fmt.Sprintf("%s:%d", detectLocalIP(), *rayPort)
	}

	server := NewServer(*minNodes, *modelName, resolvedRayHead)
	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("starting coordinator on %s", addr)
	log.Fatal(http.ListenAndServe(addr, server.Handler()))
}
