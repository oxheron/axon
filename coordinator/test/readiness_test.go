package server_test

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"axon/coordinator/internal/server"
)

func TestBuildStatusExecutablePipelineReadiness(t *testing.T) {
	healthServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer healthServer.Close()

	srv := server.NewServer(2, "test-model", "vllm_slice", server.BackendConfig{})
	startup := &server.StartupConfig{
		ClusterID:            "cluster-test",
		ModelName:            "test-model",
		ExecutionMode:        "vllm_slice",
		PipelineParallelSize: 2,
		StageCount:           2,
		EntryNodeID:          "node-a",
	}
	srv.ApplyTestClusterState(
		"cluster-test",
		[]string{"node-a", "node-b"},
		map[string]server.NodeInfo{
			"node-a": {
				NodeID:      "node-a",
				Host:        "127.0.0.1",
				Port:        9000,
				WorkerURL:   healthServer.URL,
			},
			"node-b": {
				NodeID:      "node-b",
				Host:        "127.0.0.1",
				Port:        9001,
				WorkerURL:   "http://127.0.0.1:8101",
			},
		},
		map[string]server.NodeAssignment{
			"node-a": {
				NodeID:       "node-a",
				StageIndex:   0,
				StageCount:   2,
				StageRole:    "entry",
				LoadStrategy: "vllm_slice",
			},
			"node-b": {
				NodeID:       "node-b",
				StageIndex:   1,
				StageCount:   2,
				StageRole:    "final",
				LoadStrategy: "vllm_slice",
			},
		},
		map[string]server.NodeRuntimeStatus{
			"node-a": {
				NodeID:         "node-a",
				ClusterID:      "cluster-test",
				LifecycleState: "pipeline_ready",
				WorkerURL:      healthServer.URL,
				UpdatedAt:      time.Now().UTC(),
			},
			"node-b": {
				NodeID:         "node-b",
				ClusterID:      "cluster-test",
				LifecycleState: "pipeline_ready",
				WorkerURL:      "http://127.0.0.1:8101",
				UpdatedAt:      time.Now().UTC(),
			},
		},
		startup,
	)

	status := srv.BuildStatus()

	if !status.ClusterReady || !status.AllNodesReady || !status.EntryNodeReady || !status.BackendReady {
		t.Fatalf("expected cluster/all/entry/backend readiness, got %+v", status)
	}
	if !status.PipelineReady || !status.InferenceReady {
		t.Fatalf("expected pipeline and inference readiness, got %+v", status)
	}
}

func TestBuildStatusDryRunIsPipelineReadyButNotInferenceReady(t *testing.T) {
	srv := server.NewServer(2, "test-model", "dry_run", server.BackendConfig{})
	startup := &server.StartupConfig{
		ClusterID:            "cluster-test",
		ModelName:            "test-model",
		ExecutionMode:        "dry_run",
		PipelineParallelSize: 2,
		StageCount:           2,
		EntryNodeID:          "node-a",
	}
	srv.ApplyTestClusterState(
		"cluster-test",
		[]string{"node-a", "node-b"},
		map[string]server.NodeInfo{
			"node-a": {
				NodeID:      "node-a",
				Host:        "127.0.0.1",
				Port:        9000,
				WorkerURL:   "http://127.0.0.1:8100",
			},
			"node-b": {
				NodeID:      "node-b",
				Host:        "127.0.0.1",
				Port:        9001,
				WorkerURL:   "http://127.0.0.1:8101",
			},
		},
		nil,
		map[string]server.NodeRuntimeStatus{
			"node-a": {
				NodeID:         "node-a",
				ClusterID:      "cluster-test",
				LifecycleState: "dry_run_ready",
				WorkerURL:      "http://127.0.0.1:8100",
				UpdatedAt:      time.Now().UTC(),
			},
			"node-b": {
				NodeID:         "node-b",
				ClusterID:      "cluster-test",
				LifecycleState: "dry_run_ready",
				WorkerURL:      "http://127.0.0.1:8101",
				UpdatedAt:      time.Now().UTC(),
			},
		},
		startup,
	)

	status := srv.BuildStatus()

	if !status.ClusterReady || !status.AllNodesReady || !status.EntryNodeReady || !status.PipelineReady {
		t.Fatalf("expected dry-run cluster to be pipeline ready, got %+v", status)
	}
	if status.BackendReady {
		t.Fatalf("expected dry-run backend_ready to be false, got %+v", status)
	}
	if status.InferenceReady {
		t.Fatalf("expected dry-run inference_ready to be false, got %+v", status)
	}
}
