package server

import "time"

type NodeRegistration struct {
	NodeID    string  `json:"node_id"`
	Host      string  `json:"host"`
	Port      int     `json:"port"`
	VRAMGB    float64 `json:"vram_gb"`
	WorkerURL string  `json:"worker_url"`
}

type PeerNode struct {
	NodeID     string `json:"node_id"`
	StageIndex int    `json:"stage_index"`
	StageRole  string `json:"stage_role"`
	WorkerURL  string `json:"worker_url"`
}

type SliceSpec struct {
	Kind           string `json:"kind"`
	StageIndex     int    `json:"stage_index"`
	StageCount     int    `json:"stage_count"`
	PartitionLabel string `json:"partition_label,omitempty"`
	Executable     bool   `json:"executable"`
}

type NodeAssignment struct {
	NodeID         string     `json:"node_id"`
	StageIndex     int        `json:"stage_index"`
	StageCount     int        `json:"stage_count"`
	StageRole      string     `json:"stage_role"`
	LoadStrategy   string     `json:"load_strategy"`
	SliceSpec      SliceSpec  `json:"slice_spec"`
	PeerNodes      []PeerNode `json:"peer_nodes,omitempty"`
	WorkerEndpoint string     `json:"worker_endpoint"`
}

type BackendConfig struct {
	EnvOverrides map[string]string `json:"env_overrides,omitempty"`
	LaunchArgs     []string          `json:"launch_args,omitempty"`
}

type StartupConfig struct {
	ClusterID            string          `json:"cluster_id,omitempty"`
	ModelName            string          `json:"model_name"`
	ExecutionMode        string          `json:"execution_mode,omitempty"`
	PipelineParallelSize int             `json:"pipeline_parallel_size"`
	StageCount           int             `json:"stage_count,omitempty"`
	EntryNodeID          string          `json:"entry_node_id,omitempty"`
	Nodes                []NodeInfo      `json:"nodes,omitempty"`
	Assignment           *NodeAssignment `json:"assignment,omitempty"`
	BackendConfig        BackendConfig   `json:"backend_config,omitempty"`
}

type NodeStatusUpdate struct {
	NodeID          string          `json:"node_id"`
	ClusterID       string          `json:"cluster_id,omitempty"`
	LifecycleState  string          `json:"lifecycle_state"`
	LifecycleDetail string          `json:"lifecycle_detail,omitempty"`
	WorkerURL       string          `json:"worker_url,omitempty"`
	Assignment      *NodeAssignment `json:"assignment,omitempty"`
}

type RegistrationResponse struct {
	Accepted         bool `json:"accepted"`
	RegisteredNodes  int  `json:"registered_nodes"`
	StartupTriggered bool `json:"startup_triggered"`
}

type NodeInfo struct {
	NodeID          string          `json:"node_id"`
	Host            string          `json:"host"`
	Port            int             `json:"port"`
	VRAMGB          float64         `json:"vram_gb"`
	WorkerURL       string          `json:"worker_url"`
	StageIndex      *int            `json:"stage_index,omitempty"`
	StageRole       string          `json:"stage_role,omitempty"`
	ClusterID       string          `json:"cluster_id,omitempty"`
	LifecycleState  string          `json:"lifecycle_state,omitempty"`
	LifecycleDetail string          `json:"lifecycle_detail,omitempty"`
	Assignment      *NodeAssignment `json:"assignment,omitempty"`
	UpdatedAt       string          `json:"updated_at,omitempty"`
}

type NodeRuntimeStatus struct {
	NodeID          string
	ClusterID       string
	LifecycleState  string
	LifecycleDetail string
	WorkerURL       string
	UpdatedAt       time.Time
}

type StartupTarget struct {
	Node   NodeInfo
	Config StartupConfig
}

// SignalMsg is sent by a node over the WebSocket channel to advertise its external address.
type SignalMsg struct {
	Type          string `json:"type"` // "signal"
	ExternalAddr  string `json:"external_addr"`
	ExternalPort  int    `json:"external_port"`
	TransportMode string `json:"transport_mode"` // "port_forward" | "hole_punch"
}

// NodeSignal stores one node's external address for inclusion in SignalReadyMsg.
type NodeSignal struct {
	NodeID        string `json:"node_id"`
	ExternalAddr  string `json:"external_addr"`
	ExternalPort  int    `json:"external_port"`
	TransportMode string `json:"transport_mode"`
}

// SignalReadyMsg is broadcast to all nodes when every peer has sent a signal.
type SignalReadyMsg struct {
	Type      string       `json:"type"` // "signal_ready"
	ClusterID string       `json:"cluster_id"`
	Peers     []NodeSignal `json:"peers"`
}

// ErrorMsg is pushed by the coordinator to indicate a signaling error.
type ErrorMsg struct {
	Type    string `json:"type"` // "error"
	Code    string `json:"code"` // "signal_timeout", "unknown_node"
	Message string `json:"message"`
}

type StatusResponse struct {
	MinNodes        int        `json:"min_nodes"`
	RegisteredNodes int        `json:"registered_nodes"`
	ClusterReady    bool       `json:"cluster_ready"`
	EntryNodeReady  bool       `json:"entry_node_ready"`
	AllNodesReady   bool       `json:"all_nodes_ready"`
	BackendReady    bool       `json:"backend_ready"`
	PipelineReady   bool       `json:"pipeline_ready"`
	InferenceReady  bool       `json:"inference_ready"`
	ClusterID       string     `json:"cluster_id,omitempty"`
	ModelName       string     `json:"model_name"`
	ExecutionMode   string     `json:"execution_mode,omitempty"`
	StageCount      int        `json:"stage_count,omitempty"`
	EntryNodeID     string     `json:"entry_node_id,omitempty"`
	SelectedNodeID  string     `json:"selected_node_id,omitempty"`
	Nodes           []NodeInfo `json:"nodes"`
}
