// Package llm is the one client every AI feature of the panel uses to talk to a
// language model (TASK-0190, ADR-0041, REQ-0109). The caller names a provider
// (Anthropic, OpenAI, Google Gemini or OpenRouter), a model and a key; each
// provider is reached through its official Go SDK, and the code that asks does not
// change with the provider. Answers and errors have one shape.
//
// The SDKs never open a connection of their own: the caller gives the HTTP client,
// which in the panel is the safe outbound client (REQ-0092). A key is used for the
// call and is never logged, returned or put in an error.
package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Provider names a service.
type Provider string

// The four providers (ADR-0041).
const (
	Anthropic  Provider = "anthropic"
	OpenAI     Provider = "openai"
	Gemini     Provider = "gemini"
	OpenRouter Provider = "openrouter"
)

// Providers lists them in the order screens show them.
var Providers = []Provider{Anthropic, OpenAI, Gemini, OpenRouter}

// Valid reports whether p is one of the four.
func (p Provider) Valid() bool {
	for _, q := range Providers {
		if p == q {
			return true
		}
	}
	return false
}

// Config makes a client.
type Config struct {
	Provider Provider
	APIKey   string
	Model    string
	// BaseURL replaces the provider's address; tests point it at a fake.
	BaseURL string
	// HTTPClient carries every request. It is required: the SDKs must not use a
	// client the panel did not give them.
	HTTPClient *http.Client
}

// Roles of a message.
const (
	User      = "user"
	Assistant = "assistant"
)

// Message is one turn of the conversation, as text.
type Message struct {
	Role string // User or Assistant
	Text string
}

// Request is what is asked of the model.
type Request struct {
	System      string
	Messages    []Message // alternating, starting with a user turn
	MaxTokens   int       // the most the model may write; required
	Temperature *float64  // nil: the provider's default
}

// Usage counts tokens.
type Usage struct {
	Input  int
	Output int
}

// Stop reasons, the same for every provider.
const (
	StopEnd     = "end"     // the model finished
	StopLength  = "length"  // it reached MaxTokens
	StopRefused = "refused" // it declined, or the provider's safety filter did
	StopOther   = "other"
)

// Response is the model's answer.
type Response struct {
	Text  string
	Stop  string
	Model string
	Usage Usage
}

// Chunk is a piece of text of an answer being written.
type Chunk struct{ Text string }

// Client talks to one provider with one key and one model.
type Client interface {
	Provider() Provider
	// Complete asks and waits for the whole answer.
	Complete(ctx context.Context, r Request) (Response, error)
	// Stream asks and calls emit with each piece as it arrives; it returns the
	// whole answer. A non-nil error from emit stops the stream and is returned.
	Stream(ctx context.Context, r Request, emit func(Chunk) error) (Response, error)
}

// New returns a client for the configured provider.
func New(c Config) (Client, error) {
	switch {
	case !c.Provider.Valid():
		return nil, fmt.Errorf("llm: %q is not a provider (use anthropic, openai, gemini or openrouter)", c.Provider)
	case strings.TrimSpace(c.APIKey) == "":
		return nil, errors.New("llm: an API key is required")
	case strings.TrimSpace(c.Model) == "":
		return nil, errors.New("llm: a model is required")
	case c.HTTPClient == nil:
		return nil, errors.New("llm: an HTTP client is required")
	}
	switch c.Provider {
	case Anthropic:
		return newAnthropic(c)
	case OpenAI:
		return newOpenAI(c)
	case Gemini:
		return newGemini(c)
	default:
		return newOpenRouter(c)
	}
}

// Check reports what is wrong with a request before any provider is asked.
func (r Request) Check() error {
	if r.MaxTokens <= 0 {
		return &Error{Kind: ErrBadRequest, Message: "the token limit must be above zero"}
	}
	if len(r.Messages) == 0 || r.Messages[0].Role != User {
		return &Error{Kind: ErrBadRequest, Message: "the first message must be from the user"}
	}
	for i, m := range r.Messages {
		if m.Role != User && m.Role != Assistant {
			return &Error{Kind: ErrBadRequest, Message: fmt.Sprintf("message %d has the role %q", i+1, m.Role)}
		}
		if strings.TrimSpace(m.Text) == "" {
			return &Error{Kind: ErrBadRequest, Message: fmt.Sprintf("message %d is empty", i+1)}
		}
		if i > 0 && m.Role == r.Messages[i-1].Role {
			return &Error{Kind: ErrBadRequest, Message: "user and assistant messages must alternate"}
		}
	}
	return nil
}

// Kind classifies an error the same way for every provider.
type Kind string

// Error kinds.
const (
	ErrAuth        Kind = "auth"        // the key was refused or lacks the right
	ErrRateLimit   Kind = "rate_limit"  // too many requests, or the quota is used up
	ErrTooLarge    Kind = "too_large"   // the request is more than the model takes
	ErrBadRequest  Kind = "bad_request" // the request or the model name is wrong
	ErrUnavailable Kind = "unavailable" // the provider is down, overloaded or unreachable
	ErrCanceled    Kind = "canceled"    // the caller stopped it
	ErrInternal    Kind = "internal"    // anything else
)

// Error is a failure of a call. Its text never holds the key or the provider's body.
type Error struct {
	Kind     Kind
	Provider Provider
	Status   int    // the HTTP status, when there was one
	Message  string // a short reason in words
}

func (e *Error) Error() string {
	p := string(e.Provider)
	if p == "" {
		p = "llm"
	}
	return fmt.Sprintf("%s: %s: %s", p, e.Kind, e.Message)
}

// Is lets errors.Is match on the kind: errors.Is(err, &llm.Error{Kind: llm.ErrRateLimit}).
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Kind == e.Kind && (t.Provider == "" || t.Provider == e.Provider)
}

// kindOfStatus maps an HTTP status to a kind.
func kindOfStatus(status int) (Kind, string) {
	switch {
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return ErrAuth, "the key was refused"
	case status == http.StatusTooManyRequests:
		return ErrRateLimit, "the rate limit or quota was reached"
	case status == http.StatusRequestEntityTooLarge:
		return ErrTooLarge, "the request is too large"
	case status == http.StatusBadRequest, status == http.StatusNotFound, status == http.StatusUnprocessableEntity:
		return ErrBadRequest, "the provider did not accept the request or the model"
	case status >= 500, status == 529:
		return ErrUnavailable, "the provider is not available"
	}
	return ErrInternal, fmt.Sprintf("the provider answered %d", status)
}

// fail turns an error from an SDK into an *Error. status is 0 when the SDK gave
// none (a network failure); the provider's own words are dropped.
func fail(p Provider, ctx context.Context, status int, err error) error {
	if err == nil {
		return nil
	}
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return &Error{Kind: ErrCanceled, Provider: p, Message: "the call was stopped"}
	}
	if status == 0 {
		return &Error{Kind: ErrUnavailable, Provider: p, Message: "the provider could not be reached"}
	}
	k, msg := kindOfStatus(status)
	return &Error{Kind: k, Provider: p, Status: status, Message: msg}
}
