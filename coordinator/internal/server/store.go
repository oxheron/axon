package server

import (
	"bytes"
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

// compareSet implements torch.distributed.Store::compare_set semantics.
func (cs *clusterStore) compareSet(key string, expected, desired []byte) []byte {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	expiry := time.Now().Add(10 * time.Minute)
	e, ok := cs.entries[key]
	if !ok {
		if len(expected) == 0 {
			v := append([]byte(nil), desired...)
			cs.entries[key] = &kvEntry{value: v, expiry: expiry}
			cs.cond.Broadcast()
			return v
		}
		return []byte{}
	}
	cur := e.value
	if len(expected) == 0 && len(cur) == 0 {
		v := append([]byte(nil), desired...)
		e.value = v
		e.expiry = expiry
		cs.cond.Broadcast()
		return v
	}
	if bytes.Equal(cur, expected) {
		v := append([]byte(nil), desired...)
		e.value = v
		e.expiry = expiry
		cs.cond.Broadcast()
		return v
	}
	return append([]byte(nil), cur...)
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

// handleStore routes (key may contain slashes; path is /store/{cluster_id}/{rest}):
//
//	PUT    /store/{cluster_id}/{key}                body: {"value":"<base64>"}
//	GET    /store/{cluster_id}/{key}?timeout_ms    body: {"value":"<base64>"}
//	POST   /store/{cluster_id}/{key}/add            body: {"amount":N}  → {"value":N}
//	POST   /store/{cluster_id}/{key}/compare_set    body: {"expected":"<b64>","desired":"<b64>"} → {"value":"<b64>"}
//	DELETE /store/{cluster_id}                       cleans up the whole cluster namespace
func (s *Server) handleStore(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/store/")
	if path == "" || strings.HasPrefix(path, "/") {
		http.Error(w, "missing cluster_id", http.StatusBadRequest)
		return
	}

	// DELETE /store/{cluster_id} — no key segment
	if r.Method == http.MethodDelete && !strings.Contains(path, "/") {
		s.store.deleteCluster(path)
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
		return
	}

	idx := strings.IndexByte(path, '/')
	if idx <= 0 || idx >= len(path)-1 {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}
	clusterID := path[:idx]
	rest := path[idx+1:]
	if rest == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	var key string
	if strings.HasSuffix(rest, "/add") {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		key = strings.TrimSuffix(rest, "/add")
		var body struct {
			Amount int64 `json:"amount"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		cs := s.store.cluster(clusterID)
		result := cs.add(key, body.Amount)
		writeJSON(w, http.StatusOK, map[string]int64{"value": result})
		log.Printf("[store] add cluster=%s key=%s amount=%d -> %d", clusterID, key, body.Amount, result)
		return
	}
	if strings.HasSuffix(rest, "/compare_set") {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		key = strings.TrimSuffix(rest, "/compare_set")
		var body struct {
			Expected string `json:"expected"`
			Desired  string `json:"desired"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		exp, err := base64.StdEncoding.DecodeString(body.Expected)
		if err != nil {
			http.Error(w, "bad expected base64", http.StatusBadRequest)
			return
		}
		des, err := base64.StdEncoding.DecodeString(body.Desired)
		if err != nil {
			http.Error(w, "bad desired base64", http.StatusBadRequest)
			return
		}
		cs := s.store.cluster(clusterID)
		out := cs.compareSet(key, exp, des)
		encoded := base64.StdEncoding.EncodeToString(out)
		writeJSON(w, http.StatusOK, map[string]string{"value": encoded})
		log.Printf("[store] compare_set cluster=%s key=%s", clusterID, key)
		return
	}
	key = rest
	cs := s.store.cluster(clusterID)

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
