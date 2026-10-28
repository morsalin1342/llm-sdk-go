package llm

import (
	"context"
	"net/http"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/optionalnullable"
	"github.com/OpenRouterTeam/go-sdk/retry"
)

type openrouterClient struct {
	c     *openrouter.OpenRouter
	model string
}

// statusKey carries a place for the HTTP status of one call, because the SDK
// has a different error type for each status and the status is what we need.
type statusKey struct{}

// statusClient is the HTTP client the SDK is given: it passes every request to the
// caller's client and notes the status of the answer.
type statusClient struct{ inner *http.Client }

func (s statusClient) Do(req *http.Request) (*http.Response, error) {
	res, err := s.inner.Do(req)
	if res != nil {
		if p, ok := req.Context().Value(statusKey{}).(*int); ok {
			*p = res.StatusCode
		}
	}
	return res, err
}

func newOpenRouter(cfg Config) (Client, error) {
	opts := []openrouter.SDKOption{openrouter.WithSecurity(cfg.APIKey), openrouter.WithClient(statusClient{cfg.HTTPClient}),
		openrouter.WithRetryConfig(retry.Config{Strategy: "none"})}
	if cfg.BaseURL != "" {
		opts = append(opts, openrouter.WithServerURL(cfg.BaseURL))
	}
	return &openrouterClient{c: openrouter.New(opts...), model: cfg.Model}, nil
}

func (o *openrouterClient) Provider() Provider { return OpenRouter }

func (o *openrouterClient) request(r Request, stream bool) components.ChatRequest {
	var msgs []components.ChatMessages
	if r.System != "" {
		msgs = append(msgs, components.CreateChatMessagesSystem(components.ChatSystemMessage{
			Content: components.CreateChatSystemMessageContentStr(r.System), Role: components.ChatSystemMessageRoleSystem}))
	}
	for _, m := range r.Messages {
		if m.Role == Assistant {
			text := m.Text
			msgs = append(msgs, components.CreateChatMessagesAssistant(components.ChatAssistantMessage{
				Content: optionalnullable.From(ptr(components.CreateChatAssistantMessageContentStr(text))), Role: components.ChatAssistantMessageRoleAssistant}))
		} else {
			msgs = append(msgs, components.CreateChatMessagesUser(components.ChatUserMessage{
				Content: components.CreateChatUserMessageContentStr(m.Text), Role: components.ChatUserMessageRoleUser}))
		}
	}
	req := components.ChatRequest{Model: ptr(o.model), Messages: msgs, Stream: ptr(stream),
		MaxCompletionTokens: optionalnullable.From(ptr(int64(r.MaxTokens)))}
	if r.Temperature != nil {
		req.Temperature = optionalnullable.From(ptr(*r.Temperature))
	}
	return req
}

func ptr[T any](v T) *T { return &v }

func openrouterStop(r *components.ChatFinishReasonEnum) string {
	if r == nil {
		return StopOther
	}
	return openaiStop(string(*r))
}

func (o *openrouterClient) Complete(ctx context.Context, r Request) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	var status int
	ctx = context.WithValue(ctx, statusKey{}, &status)
	res, err := o.c.Chat.Send(ctx, o.request(r, false), nil)
	if err != nil {
		return Response{}, fail(OpenRouter, ctx, status, err)
	}
	if res == nil || res.ChatResult == nil {
		return Response{}, &Error{Kind: ErrInternal, Provider: OpenRouter, Message: "the answer had no result"}
	}
	out := Response{Model: res.ChatResult.Model}
	if u := res.ChatResult.Usage; u != nil {
		out.Usage = Usage{Input: int(u.PromptTokens), Output: int(u.CompletionTokens)}
	}
	if len(res.ChatResult.Choices) > 0 {
		c := res.ChatResult.Choices[0]
		out.Stop = openrouterStop(c.FinishReason)
		if content, ok := c.Message.Content.GetOrZero(); ok && content.Str != nil {
			out.Text = *content.Str
		}
	}
	return out, nil
}

func (o *openrouterClient) Stream(ctx context.Context, r Request, emit func(Chunk) error) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	var status int
	ctx = context.WithValue(ctx, statusKey{}, &status)
	res, err := o.c.Chat.Send(ctx, o.request(r, true), nil)
	if err != nil {
		return Response{}, fail(OpenRouter, ctx, status, err)
	}
	if res == nil || res.EventStream == nil {
		return Response{}, &Error{Kind: ErrInternal, Provider: OpenRouter, Message: "the answer was not a stream"}
	}
	defer res.EventStream.Close()
	out := Response{Model: o.model, Stop: StopOther}
	for res.EventStream.Next() {
		ev := res.EventStream.Value()
		if ev == nil {
			continue
		}
		d := ev.Data
		if d.Model != "" {
			out.Model = d.Model
		}
		if d.Usage != nil {
			out.Usage = Usage{Input: int(d.Usage.PromptTokens), Output: int(d.Usage.CompletionTokens)}
		}
		if len(d.Choices) == 0 {
			continue
		}
		if d.Choices[0].FinishReason != nil {
			out.Stop = openrouterStop(d.Choices[0].FinishReason)
		}
		if text, ok := d.Choices[0].Delta.Content.GetOrZero(); ok && text != "" {
			out.Text += text
			if err := emit(Chunk{Text: text}); err != nil {
				return Response{}, err
			}
		}
	}
	if err := res.EventStream.Err(); err != nil {
		return Response{}, fail(OpenRouter, ctx, status, err)
	}
	return out, nil
}
