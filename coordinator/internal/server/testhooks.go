package server

// ApplyTestClusterState replaces selected coordinator fields for unit tests.
// It is not used by the coordinator at runtime; do not call from production code.
func (s *Server) ApplyTestClusterState(
	clusterID string,
	nodeOrder []string,
	nodes map[string]NodeInfo,
	assignments map[string]NodeAssignment,
	nodeRuntime map[string]NodeRuntimeStatus,
	startupConfig *StartupConfig,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if clusterID != "" {
		s.clusterID = clusterID
	}
	if nodeOrder != nil {
		s.nodeOrder = append([]string(nil), nodeOrder...)
	}
	if nodes != nil {
		s.nodes = cloneNodeMap(nodes)
	}
	if assignments != nil {
		s.assignments = cloneAssignmentMap(assignments)
	}
	if nodeRuntime != nil {
		s.nodeRuntime = cloneRuntimeMap(nodeRuntime)
	}
	s.startupConfig = startupConfig
}

func cloneNodeMap(in map[string]NodeInfo) map[string]NodeInfo {
	out := make(map[string]NodeInfo, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneAssignmentMap(in map[string]NodeAssignment) map[string]NodeAssignment {
	out := make(map[string]NodeAssignment, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneRuntimeMap(in map[string]NodeRuntimeStatus) map[string]NodeRuntimeStatus {
	out := make(map[string]NodeRuntimeStatus, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
