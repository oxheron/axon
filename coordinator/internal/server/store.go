package server

import (
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// kvEntry holds a stored value and its expiry time.
type kvEntry struct {
	value   []byte
	counter int64 // used for atomic add
	expiry  time.Time
}

// clusterStore is the per-cluster KV namespace.
type clusterStore struct {
	mu      sync.Mutex
	cond    *sync.Cond
	entries map[string]*kvEntry
}

func newClusterStore() *clusterStore {
	cs := &clusterStore{entries: make(map[string]*kvEntry)}
	cs.cond = sync.NewCond(&cs.mu)
	return cs
}

func (cs *clusterStore) set(key string, value []byte) {
	cs.mu.Lock()
	cs.entries[key] = &kvEntry{
		value:  value,
		expiry: time.Now().Add(10 * time.Minute),
	}
	cs.cond.Broadcast()
	cs.mu.Unlock()
}

// get blocks until the key exists or the deadline is reached.
func (cs *clusterStore) get(key string, timeout time.Duration) ([]byte, bool) {
	deadline := time.Now().Add(timeout)
	cs.mu.Lock()
	defer cs.mu.Unlock()
	for {
		if e, ok := cs.entries[key]; ok && time.Now().Before(e.expiry) {
			return e.value, true
		}
		if time.Now().After(deadline) {
			return nil, false
		}
		// Wait with a short sleep so the deadline check fires.
		timer := time.AfterFunc(100*time.Millisecond, func() { cs.cond.Broadcast() })
		cs.cond.Wait()
		timer.Stop()
	}
}

func (cs *clusterStore) add(key string, amount int64) int64 {
	cs.mu.Lock()
	e, ok := cs.entries[key]
	if !ok {
		e = &kvEntry{expiry: time.Now().Add(10 * time.Minute)}
		cs.entries[key] = e
	}
	e.counter += amount
	result := e.counter
	cs.cond.Broadcast()
	cs.mu.Unlock()
	return result
}

func (cs *clusterStore) deleteKey(key string) bool {
	cs.mu.Lock()
	_, ok := cs.entries[key]
	if ok {
		delete(cs.entries, key)
	}
	cs.mu.Unlock()
	return ok
}

func (cs *clusterStore) numKeys() int {
	cs.mu.Lock()
	n := len(cs.entries)
	cs.mu.Unlock()
	return n
}

// ---- Server-level store registry ----

type storeRegistry struct {
	mu       sync.Mutex
	clusters map[string]*clusterStore
}

func newStoreRegistry() *storeRegistry {
	return &storeRegistry{clusters: make(map[string]*clusterStore)}
}

func (r *storeRegistry) cluster(id string) *clusterStore {
	r.mu.Lock()
	cs, ok := r.clusters[id]
	if !ok {
		cs = newClusterStore()
		r.clusters[id] = cs
	}
	r.mu.Unlock()
	return cs
}

func (r *storeRegistry) deleteCluster(id string) {
	r.mu.Lock()
	delete(r.clusters, id)
	r.mu.Unlock()
}

// ---- HTTP handler ----

// handleStore routes:
//
//	PUT    /store/{cluster_id}/{key}           body: {"value":"<base64>"}
//	GET    /store/{cluster_id}/{key}?timeout_ms body: {"value":"<base64>"}
//	POST   /store/{cluster_id}/{key}/add        body: {"amount":N}  → {"value":N}
//	DELETE /store/{cluster_id}                  cleans up the whole cluster namespace
func (s *Server) handleStore(w http.ResponseWriter, r *http.Request) {
	// Strip leading /store/
	path := strings.TrimPrefix(r.URL.Path, "/store/")
	parts := strings.SplitN(path, "/", 3)

	if len(parts) < 1 || parts[0] == "" {
		http.Error(w, "missing cluster_id", http.StatusBadRequest)
		return
	}
	clusterID := parts[0]

	// DELETE /store/{cluster_id} — clean up cluster namespace
	if r.Method == http.MethodDelete && len(parts) == 1 {
		s.store.deleteCluster(clusterID)
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
		return
	}

	if len(parts) < 2 || parts[1] == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}
	key := parts[1]
	cs := s.store.cluster(clusterID)

	// POST /store/{cluster_id}/{key}/add
	if r.Method == http.MethodPost && len(parts) == 3 && parts[2] == "add" {
		var body struct {
			Amount int64 `json:"amount"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		result := cs.add(key, body.Amount)
		writeJSON(w, http.StatusOK, map[string]int64{"value": result})
		log.Printf("[store] add cluster=%s key=%s amount=%d -> %d", clusterID, key, body.Amount, result)
		return
	}

	switch r.Method {
	case http.MethodPut:
		var body struct {
			Value string `json:"value"` // base64-encoded bytes
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		raw, err := base64.StdEncoding.DecodeString(body.Value)
		if err != nil {
			http.Error(w, "bad base64", http.StatusBadRequest)
			return
		}
		cs.set(key, raw)
		log.Printf("[store] set cluster=%s key=%s len=%d", clusterID, key, len(raw))
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})

	case http.MethodGet:
		timeoutMs := 30000
		if v := r.URL.Query().Get("timeout_ms"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				timeoutMs = n
			}
		}
		timeout := time.Duration(timeoutMs) * time.Millisecond
		val, ok := cs.get(key, timeout)
		if !ok {
			http.Error(w, "key not found or timeout", http.StatusRequestTimeout)
			return
		}
		encoded := base64.StdEncoding.EncodeToString(val)
		writeJSON(w, http.StatusOK, map[string]string{"value": encoded})

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}
