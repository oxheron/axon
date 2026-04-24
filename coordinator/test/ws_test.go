package server_test

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
	"axon/coordinator/internal/server"
)

func wsURLFor(ts *httptest.Server) string {
	return strings.Replace(ts.URL, "http://", "ws://", 1)
}

func twoNodeServer(t *testing.T) (*server.Server, *httptest.Server) {
	t.Helper()
	srv := server.NewServer(2, "test-model", "", "dry_run", server.BackendConfig{})
	startup := &server.StartupConfig{
		ClusterID:            "cluster-ws-test",
		ModelName:            "test-model",
		ExecutionMode:        "dry_run",
		PipelineParallelSize: 2,
		StageCount:           2,
		EntryNodeID:          "node-a",
	}
	srv.ApplyTestClusterState(
		"cluster-ws-test",
		[]string{"node-a", "node-b"},
		map[string]server.NodeInfo{
			"node-a": {NodeID: "node-a", Host: "127.0.0.1", Port: 9000, CallbackURL: "http://127.0.0.1:9000"},
			"node-b": {NodeID: "node-b", Host: "127.0.0.1", Port: 9001, CallbackURL: "http://127.0.0.1:9001"},
		},
		map[string]server.NodeAssignment{
			"node-a": {NodeID: "node-a", StageIndex: 0, StageCount: 2, StageRole: "entry"},
			"node-b": {NodeID: "node-b", StageIndex: 1, StageCount: 2, StageRole: "final"},
		},
		nil,
		startup,
	)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ts
}

func TestWSHandlerRejectsMissingNodeID(t *testing.T) {
	_, ts := twoNodeServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws", nil)
	if err == nil {
		t.Fatal("expected rejection for missing node_id")
	}
}

func TestWSHandlerRejectsUnknownNodeID(t *testing.T) {
	_, ts := twoNodeServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=ghost", nil)
	if err == nil {
		t.Fatal("expected rejection for unknown node_id")
	}
}

func TestWSHandlerRejectsWrongToken(t *testing.T) {
	srv := server.NewServer(2, "test-model", "", "", server.BackendConfig{})
	srv.SetSignalingOptions("sekret", 10*time.Second)
	srv.ApplyTestClusterState("", []string{"node-a"}, map[string]server.NodeInfo{
		"node-a": {NodeID: "node-a", Host: "127.0.0.1", Port: 9000},
	}, nil, nil, nil)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-a&token=wrong", nil)
	if err == nil {
		t.Fatal("expected rejection for wrong token")
	}
}

func TestWSHandlerAcceptsValidTokenAndPushesStartupConfig(t *testing.T) {
	srv := server.NewServer(2, "test-model", "", "dry_run", server.BackendConfig{})
	srv.SetSignalingOptions("sekret", 10*time.Second)
	startup := &server.StartupConfig{
		ClusterID: "cluster-tok", ModelName: "test-model",
		ExecutionMode: "dry_run", StageCount: 1, EntryNodeID: "node-a",
	}
	srv.ApplyTestClusterState("cluster-tok", []string{"node-a"},
		map[string]server.NodeInfo{"node-a": {NodeID: "node-a", Host: "127.0.0.1", Port: 9000}},
		nil, nil, startup,
	)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-a&token=sekret", nil)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	var msg map[string]any
	if err := wsjson.Read(ctx, conn, &msg); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if msg["type"] != wsMsgTypeStartupConfig {
		t.Fatalf("expected startup_config, got %v", msg["type"])
	}
	if msg["cluster_id"] != "cluster-tok" {
		t.Fatalf("unexpected cluster_id: %v", msg["cluster_id"])
	}
}

func TestWSHandlerPushesStartupConfigOnConnect(t *testing.T) {
	_, ts := twoNodeServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-a", nil)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	var msg map[string]any
	if err := wsjson.Read(ctx, conn, &msg); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if msg["type"] != wsMsgTypeStartupConfig {
		t.Fatalf("expected startup_config, got %v", msg["type"])
	}
	if msg["cluster_id"] != "cluster-ws-test" {
		t.Fatalf("unexpected cluster_id: %v", msg["cluster_id"])
	}
}

func TestWSHandlerSignalReadyBroadcastWhenBothNodesSignal(t *testing.T) {
	_, ts := twoNodeServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	connA, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-a", nil)
	if err != nil {
		t.Fatalf("node-a dial: %v", err)
	}
	defer connA.Close(websocket.StatusNormalClosure, "")

	connB, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-b", nil)
	if err != nil {
		t.Fatalf("node-b dial: %v", err)
	}
	defer connB.Close(websocket.StatusNormalClosure, "")

	// Drain startup_config pushes from both connections.
	var tmp map[string]any
	if err := wsjson.Read(ctx, connA, &tmp); err != nil {
		t.Fatalf("node-a startup_config drain: %v", err)
	}
	if err := wsjson.Read(ctx, connB, &tmp); err != nil {
		t.Fatalf("node-b startup_config drain: %v", err)
	}

	// Both nodes send their signals.
	sigA := map[string]any{"type": wsMsgTypeSignal, "external_addr": "1.2.3.4", "external_port": 12000, "transport_mode": "port_forward"}
	sigB := map[string]any{"type": wsMsgTypeSignal, "external_addr": "5.6.7.8", "external_port": 13000, "transport_mode": "hole_punch"}
	if err := wsjson.Write(ctx, connA, sigA); err != nil {
		t.Fatalf("node-a signal write: %v", err)
	}
	if err := wsjson.Write(ctx, connB, sigB); err != nil {
		t.Fatalf("node-b signal write: %v", err)
	}

	// Both should receive signal_ready.
	var readyA, readyB map[string]any
	if err := wsjson.Read(ctx, connA, &readyA); err != nil {
		t.Fatalf("node-a signal_ready read: %v", err)
	}
	if err := wsjson.Read(ctx, connB, &readyB); err != nil {
		t.Fatalf("node-b signal_ready read: %v", err)
	}

	if readyA["type"] != wsMsgTypeSignalReady {
		t.Fatalf("expected signal_ready for node-a, got %v", readyA["type"])
	}
	if readyB["type"] != wsMsgTypeSignalReady {
		t.Fatalf("expected signal_ready for node-b, got %v", readyB["type"])
	}
	if readyA["cluster_id"] != "cluster-ws-test" {
		t.Fatalf("unexpected cluster_id in signal_ready: %v", readyA["cluster_id"])
	}
	peers, _ := readyA["peers"].([]any)
	if len(peers) != 2 {
		t.Fatalf("expected 2 peers in signal_ready, got %d", len(peers))
	}
}

func TestWSHandlerSignalTimeoutSendsError(t *testing.T) {
	srv := server.NewServer(2, "test-model", "", "dry_run", server.BackendConfig{})
	srv.SetSignalingOptions("", 100*time.Millisecond) // very short timeout
	startup := &server.StartupConfig{
		ClusterID: "cluster-to", ModelName: "test-model",
		ExecutionMode: "dry_run", StageCount: 2, EntryNodeID: "node-a",
	}
	srv.ApplyTestClusterState("cluster-to",
		[]string{"node-a", "node-b"},
		map[string]server.NodeInfo{
			"node-a": {NodeID: "node-a", Host: "127.0.0.1", Port: 9000},
			"node-b": {NodeID: "node-b", Host: "127.0.0.1", Port: 9001},
		},
		nil, nil, startup,
	)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	connA, _, err := websocket.Dial(ctx, wsURLFor(ts)+"/ws?node_id=node-a", nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer connA.Close(websocket.StatusNormalClosure, "")

	// Drain startup_config.
	var tmp map[string]any
	_ = wsjson.Read(ctx, connA, &tmp)

	// Only node-a signals; node-b never does → timeout fires.
	sig := map[string]any{"type": wsMsgTypeSignal, "external_addr": "1.2.3.4", "external_port": 12000, "transport_mode": "port_forward"}
	if err := wsjson.Write(ctx, connA, sig); err != nil {
		t.Fatalf("signal write: %v", err)
	}

	var errMsg map[string]any
	if err := wsjson.Read(ctx, connA, &errMsg); err != nil {
		t.Fatalf("error msg read: %v", err)
	}
	if errMsg["type"] != wsMsgTypeError {
		t.Fatalf("expected error, got %v", errMsg["type"])
	}
	if errMsg["code"] != "signal_timeout" {
		t.Fatalf("expected signal_timeout, got %v", errMsg["code"])
	}
}

// wsMsgType* constants referenced in tests — must match ws.go.
const (
	wsMsgTypeStartupConfig = "startup_config"
	wsMsgTypeSignalReady   = "signal_ready"
	wsMsgTypeError         = "error"
	wsMsgTypeSignal        = "signal"
)
