package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/morsalin1342/llm-sdk-go"
)

// The same behaviour tests run against every adapter. Each provider has a fake of its
// API in the provider's own wire format (no key, no network), so the SDK under
// the adapter really parses what it parses in production.

const testKey = "sk-test-KEY-0123456789"

type seen struct {
	mu      sync.Mutex
	path    string
	body    string
	headers http.Header
}

func (s *seen) set(r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.path, s.body, s.headers = r.URL.Path, string(b), r.Header.Clone()
}

// fake is one provider's API: it answers a normal call, a streamed call, or a
// fixed HTTP status.
type fake struct {
	name     llm.Provider
	model    string
	baseSuf  string // appended to the server URL to make the base URL
	keyCheck func(h http.Header) bool
	ok       func(w http.ResponseWriter)
	stream   func(w http.ResponseWriter)
	wantsStr func(body string) bool // whether the request asked for a stream
}

func sse(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	f, _ := w.(http.Flusher)
	for _, e := range events {
		fmt.Fprint(w, e)
		if f != nil {
			f.Flush()
		}
	}
}

var fakes = []fake{
	{
		name: llm.Anthropic, model: "claude-test", baseSuf: "",
		keyCheck: func(h http.Header) bool { return h.Get("X-Api-Key") == testKey },
		ok: func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"Hello "},{"type":"text","text":"there"}],"stop_reason":"end_turn","usage":{"input_tokens":11,"output_tokens":4}}`)
		},
		stream: func(w http.ResponseWriter) {
			sse(w,
				"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":11,\"output_tokens\":1}}}\n\n",
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n",
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}\n\n",
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		},
		wantsStr: func(b string) bool { return strings.Contains(b, `"stream":true`) },
	},
	{
		name: llm.OpenAI, model: "gpt-test", baseSuf: "/v1/",
		keyCheck: func(h http.Header) bool { return h.Get("Authorization") == "Bearer "+testKey },
		ok: func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"Hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`)
		},
		stream: func(w http.ResponseWriter) {
			sse(w,
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "},"finish_reason":null}]}`+"\n\n",
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[{"index":0,"delta":{"content":"there"},"finish_reason":null}]}`+"\n\n",
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n",
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`+"\n\n",
				"data: [DONE]\n\n")
		},
		wantsStr: func(b string) bool { return strings.Contains(b, `"stream":true`) },
	},
	{
		name: llm.Gemini, model: "gemini-test", baseSuf: "",
		keyCheck: func(h http.Header) bool { return h.Get("X-Goog-Api-Key") == testKey },
		ok: func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "},{"text":"there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4},"modelVersion":"gemini-test-001"}`)
		},
		stream: func(w http.ResponseWriter) {
			sse(w,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "}]}}],"modelVersion":"gemini-test-001"}`+"\n\n",
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4}}`+"\n\n")
		},
		wantsStr: func(string) bool { return false }, // Gemini streams on another path
	},
	{
		name: llm.OpenRouter, model: "vendor/model-test", baseSuf: "/api/v1",
		keyCheck: func(h http.Header) bool { return h.Get("Authorization") == "Bearer "+testKey },
		ok: func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"vendor/model-test","choices":[{"index":0,"message":{"role":"assistant","content":"Hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`)
		},
		stream: func(w http.ResponseWriter) {
			sse(w,
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"vendor/model-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "},"finish_reason":null}]}`+"\n\n",
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"vendor/model-test","choices":[{"index":0,"delta":{"content":"there"},"finish_reason":null}]}`+"\n\n",
				`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"vendor/model-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`+"\n\n",
				"data: [DONE]\n\n")
		},
		wantsStr: func(b string) bool { return strings.Contains(b, `"stream":true`) },
	},
}

// server runs a fake: status 0 answers normally; otherwise it answers that status
// with a body that must never reach the caller.
func (f fake) server(t *testing.T, status int, s *seen) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.set(r)
		if status != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			fmt.Fprintf(w, `{"error":{"message":"SECRET-PROVIDER-WORDS key=%s","type":"x","code":"x"}}`, testKey)
			return
		}
		if !f.keyCheck(r.Header) {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if strings.Contains(r.URL.Path, "stream") || f.wantsStr(s.body) {
			f.stream(w)
			return
		}
		f.ok(w)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (f fake) client(t *testing.T, srv *httptest.Server) llm.Client {
	t.Helper()
	c, err := llm.New(llm.Config{Provider: f.name, APIKey: testKey, Model: f.model, BaseURL: srv.URL + f.baseSuf, HTTPClient: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func request() llm.Request {
	temp := 0.2
	return llm.Request{System: "You help with websites.", MaxTokens: 256, Temperature: &temp, Messages: []llm.Message{
		{Role: llm.User, Text: "Why is my site slow?"}, {Role: llm.Assistant, Text: "Let me look."}, {Role: llm.User, Text: "Thanks."}}}
}

func TestLLMComplete(t *testing.T) {
	for _, f := range fakes {
		t.Run(string(f.name), func(t *testing.T) {
			var s seen
			c := f.client(t, f.server(t, 0, &s))
			if c.Provider() != f.name {
				t.Fatalf("provider %s", c.Provider())
			}
			res, err := c.Complete(context.Background(), request())
			if err != nil {
				t.Fatal(err)
			}
			if res.Text != "Hello there" || res.Stop != llm.StopEnd || res.Usage.Input != 11 || res.Usage.Output != 4 {
				t.Fatalf("response: %+v", res)
			}
			if res.Model == "" {
				t.Error("no model in the response")
			}
			// What the provider was sent: the system prompt, all three turns in order, the
			// limit and the model.
			for _, want := range []string{"You help with websites.", "Why is my site slow?", "Let me look.", "Thanks.", "256"} {
				if !strings.Contains(s.body, want) {
					t.Errorf("the request lacks %q:\n%s", want, s.body)
				}
			}
			if !(strings.Index(s.body, "Why is my site slow?") < strings.Index(s.body, "Let me look.") && strings.Index(s.body, "Let me look.") < strings.Index(s.body, "Thanks.")) {
				t.Errorf("the turns are out of order:\n%s", s.body)
			}
			if f.name != llm.Gemini && !strings.Contains(s.body, f.model) {
				t.Errorf("the model is not in the request:\n%s", s.body)
			}
			if f.name == llm.Gemini && !strings.Contains(s.path, f.model) {
				t.Errorf("the model is not in the path %s", s.path)
			}
		})
	}
}

func TestLLMStream(t *testing.T) {
	for _, f := range fakes {
		t.Run(string(f.name), func(t *testing.T) {
			var s seen
			c := f.client(t, f.server(t, 0, &s))
			var got []string
			res, err := c.Stream(context.Background(), request(), func(ch llm.Chunk) error { got = append(got, ch.Text); return nil })
			if err != nil {
				t.Fatal(err)
			}
			if strings.Join(got, "|") != "Hello |there" || res.Text != "Hello there" || res.Stop != llm.StopEnd || res.Usage.Input != 11 || res.Usage.Output != 4 {
				t.Fatalf("chunks %q, response %+v", got, res)
			}
			// A caller that stops the stream gets its own error back.
			stop := errors.New("enough")
			if _, err := c.Stream(context.Background(), request(), func(llm.Chunk) error { return stop }); !errors.Is(err, stop) {
				t.Errorf("stopping the stream: %v", err)
			}
		})
	}
}

func TestLLMErrorsAreTheSameForEveryProvider(t *testing.T) {
	cases := map[int]llm.Kind{401: llm.ErrAuth, 403: llm.ErrAuth, 429: llm.ErrRateLimit, 400: llm.ErrBadRequest, 404: llm.ErrBadRequest,
		413: llm.ErrTooLarge, 500: llm.ErrUnavailable, 503: llm.ErrUnavailable}
	for _, f := range fakes {
		for status, kind := range cases {
			t.Run(fmt.Sprintf("%s %d", f.name, status), func(t *testing.T) {
				var s seen
				c := f.client(t, f.server(t, status, &s))
				for _, call := range []func() error{
					func() error { _, err := c.Complete(context.Background(), request()); return err },
					func() error {
						_, err := c.Stream(context.Background(), request(), func(llm.Chunk) error { return nil })
						return err
					},
				} {
					err := call()
					var e *llm.Error
					if !errors.As(err, &e) || e.Kind != kind || e.Provider != f.name || e.Status != status {
						t.Fatalf("error: %#v", err)
					}
					if !errors.Is(err, &llm.Error{Kind: kind}) {
						t.Errorf("errors.Is does not match the kind")
					}
					if strings.Contains(err.Error(), testKey) || strings.Contains(err.Error(), "SECRET-PROVIDER-WORDS") {
						t.Errorf("the error shows the key or the provider's words: %v", err)
					}
				}
			})
		}
	}
}

func TestLLMUnreachableAndCanceled(t *testing.T) {
	for _, f := range fakes {
		t.Run(string(f.name), func(t *testing.T) {
			var s seen
			srv := f.server(t, 0, &s)
			c := f.client(t, srv)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := c.Complete(ctx, request()); !errors.Is(err, &llm.Error{Kind: llm.ErrCanceled}) {
				t.Errorf("canceled: %v", err)
			}
			srv.Close()
			ctx2, cancel2 := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel2()
			if _, err := c.Complete(ctx2, request()); !errors.Is(err, &llm.Error{Kind: llm.ErrUnavailable}) {
				t.Errorf("unreachable: %v", err)
			}
		})
	}
}

func TestLLMRequestAndConfigAreChecked(t *testing.T) {
	good := llm.Config{Provider: llm.Anthropic, APIKey: testKey, Model: "m", HTTPClient: http.DefaultClient}
	for name, mut := range map[string]func(*llm.Config){
		"no provider":  func(c *llm.Config) { c.Provider = "" },
		"bad provider": func(c *llm.Config) { c.Provider = "cohere" },
		"no key":       func(c *llm.Config) { c.APIKey = " " },
		"no model":     func(c *llm.Config) { c.Model = "" },
		"no client":    func(c *llm.Config) { c.HTTPClient = nil },
	} {
		c := good
		mut(&c)
		if _, err := llm.New(c); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	c, err := llm.New(good)
	if err != nil {
		t.Fatal(err)
	}
	for name, r := range map[string]llm.Request{
		"no limit":                  {Messages: []llm.Message{{Role: llm.User, Text: "x"}}},
		"no messages":               {MaxTokens: 5},
		"starts with the assistant": {MaxTokens: 5, Messages: []llm.Message{{Role: llm.Assistant, Text: "x"}}},
		"empty text":                {MaxTokens: 5, Messages: []llm.Message{{Role: llm.User, Text: "  "}}},
		"not alternating":           {MaxTokens: 5, Messages: []llm.Message{{Role: llm.User, Text: "a"}, {Role: llm.User, Text: "b"}}},
		"odd role":                  {MaxTokens: 5, Messages: []llm.Message{{Role: "system", Text: "a"}}},
	} {
		if _, err := c.Complete(context.Background(), r); !errors.Is(err, &llm.Error{Kind: llm.ErrBadRequest}) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestLLMKeyGoesOnlyToTheProvider(t *testing.T) {
	for _, f := range fakes {
		t.Run(string(f.name), func(t *testing.T) {
			var s seen
			c := f.client(t, f.server(t, 0, &s))
			if _, err := c.Complete(context.Background(), request()); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(s.body, testKey) {
				t.Errorf("the key is in the request body")
			}
			if strings.Contains(s.path, testKey) {
				t.Errorf("the key is in the path %s", s.path)
			}
		})
	}
}
