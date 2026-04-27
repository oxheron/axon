package server_test

import (
	"testing"

	"axon/coordinator/internal/server"
)

func TestBuildClusterStartupConfigLocked(t *testing.T) {
	srv := server.NewServer(2, "test-model", "", server.BackendConfig{})
	srv.ApplyTestClusterState("", []string{"node-a", "node-b"}, map[string]server.NodeInfo{
		"node-a": {
			NodeID:      "node-a",
			Host:        "127.0.0.1",
			Port:        9000,
			CallbackURL: "http://127.0.0.1:9000",
			WorkerURL:   "http://127.0.0.1:8100",
		},
		"node-b": {
			NodeID:      "node-b",
			Host:        "127.0.0.1",
			Port:        9001,
			CallbackURL: "http://127.0.0.1:9001",
			WorkerURL:   "http://127.0.0.1:8101",
		},
	}, nil, nil, nil)

	cfg, targets := srv.BuildClusterStartupConfigLocked()

	if cfg.ClusterID == "" {
		t.Fatalf("expected cluster id to be populated")
	}
	if cfg.ExecutionMode != "vllm_slice" {
		t.Fatalf("expected vllm_slice execution mode, got %q", cfg.ExecutionMode)
	}
	if cfg.EntryNodeID != "node-a" {
		t.Fatalf("expected node-a as entry node, got %q", cfg.EntryNodeID)
	}
	if cfg.StageCount != 2 {
		t.Fatalf("expected stage count 2, got %d", cfg.StageCount)
	}
	if len(targets) != 2 {
		t.Fatalf("expected 2 startup targets, got %d", len(targets))
	}

	firstAssignment := targets[0].Config.Assignment
	if firstAssignment == nil {
		t.Fatalf("expected assignment for first node")
	}
	if firstAssignment.StageIndex != 0 || firstAssignment.StageRole != "entry" {
		t.Fatalf("unexpected first assignment: %+v", *firstAssignment)
	}
	if len(firstAssignment.PeerNodes) != 1 || firstAssignment.PeerNodes[0].NodeID != "node-b" {
		t.Fatalf("unexpected first peer nodes: %+v", firstAssignment.PeerNodes)
	}

	secondAssignment := targets[1].Config.Assignment
	if secondAssignment == nil {
		t.Fatalf("expected assignment for second node")
	}
	if secondAssignment.StageIndex != 1 || secondAssignment.StageRole != "final" {
		t.Fatalf("unexpected second assignment: %+v", *secondAssignment)
	}
	if len(secondAssignment.PeerNodes) != 1 || secondAssignment.PeerNodes[0].NodeID != "node-a" {
		t.Fatalf("unexpected second peer nodes: %+v", secondAssignment.PeerNodes)
	}
}

func TestBuildClusterStartupConfigLockedHonorsExplicitModeAndBackendConfig(t *testing.T) {
	srv := server.NewServer(
		2,
		"test-model",
		"dry_run",
		server.BackendConfig{
			EnvOverrides: map[string]string{
				"AXON_PHASE": "two",
			},
			LaunchArgs: []string{"--enforce-eager"},
		},
	)
	srv.ApplyTestClusterState("", []string{"node-a", "node-b"}, map[string]server.NodeInfo{
		"node-a": {
			NodeID:      "node-a",
			Host:        "127.0.0.1",
			Port:        9000,
			CallbackURL: "http://127.0.0.1:9000",
			WorkerURL:   "http://127.0.0.1:8100",
		},
		"node-b": {
			NodeID:      "node-b",
			Host:        "127.0.0.1",
			Port:        9001,
			CallbackURL: "http://127.0.0.1:9001",
			WorkerURL:   "http://127.0.0.1:8101",
		},
	}, nil, nil, nil)

	cfg, targets := srv.BuildClusterStartupConfigLocked()

	if cfg.ExecutionMode != "dry_run" {
		t.Fatalf("expected dry_run execution mode, got %q", cfg.ExecutionMode)
	}
	if cfg.BackendConfig.EnvOverrides["AXON_PHASE"] != "two" {
		t.Fatalf("missing backend env override: %+v", cfg.BackendConfig.EnvOverrides)
	}
	if len(cfg.BackendConfig.LaunchArgs) != 1 || cfg.BackendConfig.LaunchArgs[0] != "--enforce-eager" {
		t.Fatalf("unexpected backend launch args: %+v", cfg.BackendConfig.LaunchArgs)
	}
	if len(targets) != 2 {
		t.Fatalf("expected 2 startup targets, got %d", len(targets))
	}
	for _, target := range targets {
		if target.Config.Assignment == nil {
			t.Fatalf("expected assignment for node %s", target.Node.NodeID)
		}
		if target.Config.Assignment.LoadStrategy != "dry_run" {
			t.Fatalf("expected dry_run load strategy, got %q", target.Config.Assignment.LoadStrategy)
		}
		if target.Config.Assignment.SliceSpec.Executable {
			t.Fatalf("expected dry_run slice to be non-executable: %+v", target.Config.Assignment.SliceSpec)
		}
	}
}
