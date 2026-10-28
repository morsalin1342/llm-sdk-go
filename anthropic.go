package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

type anthropicClient struct {
	c     anthropic.Client
	model string
}

func newAnthropic(cfg Config) (Client, error) {
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey), option.WithHTTPClient(cfg.HTTPClient), option.WithMaxRetries(0)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &anthropicClient{c: anthropic.NewClient(opts...), model: cfg.Model}, nil
}

func (a *anthropicClient) Provider() Provider { return Anthropic }

func (a *anthropicClient) params(r Request) anthropic.MessageNewParams {
	p := anthropic.MessageNewParams{Model: anthropic.Model(a.model), MaxTokens: int64(r.MaxTokens)}
	if r.System != "" {
		p.System = []anthropic.TextBlockParam{{Text: r.System}}
	}
	for _, m := range r.Messages {
		block := anthropic.NewTextBlock(m.Text)
		if m.Role == Assistant {
			p.Messages = append(p.Messages, anthropic.NewAssistantMessage(block))
		} else {
			p.Messages = append(p.Messages, anthropic.NewUserMessage(block))
		}
	}
	if r.Temperature != nil {
		p.Temperature = anthropic.Float(*r.Temperature)
	}
	return p
}

func anthropicStop(s anthropic.StopReason) string {
	switch s {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return StopEnd
	case anthropic.StopReasonMaxTokens:
		return StopLength
	case anthropic.StopReasonRefusal:
		return StopRefused
	}
	return StopOther
}

func anthropicFail(ctx context.Context, err error) error {
	var api *anthropic.Error
	if errors.As(err, &api) {
		return fail(Anthropic, ctx, api.StatusCode, err)
	}
	return fail(Anthropic, ctx, 0, err)
}

func (a *anthropicClient) result(m *anthropic.Message) Response {
	var text strings.Builder
	for _, b := range m.Content {
		if b.Type == "text" {
			text.WriteString(b.Text)
		}
	}
	return Response{Text: text.String(), Stop: anthropicStop(m.StopReason), Model: string(m.Model),
		Usage: Usage{Input: int(m.Usage.InputTokens), Output: int(m.Usage.OutputTokens)}}
}

func (a *anthropicClient) Complete(ctx context.Context, r Request) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	m, err := a.c.Messages.New(ctx, a.params(r))
	if err != nil {
		return Response{}, anthropicFail(ctx, err)
	}
	return a.result(m), nil
}

func (a *anthropicClient) Stream(ctx context.Context, r Request, emit func(Chunk) error) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	s := a.c.Messages.NewStreaming(ctx, a.params(r))
	defer s.Close()
	var acc anthropic.Message
	for s.Next() {
		ev := s.Current()
		if err := acc.Accumulate(ev); err != nil {
			return Response{}, &Error{Kind: ErrInternal, Provider: Anthropic, Message: "the answer could not be read"}
		}
		if d, ok := ev.AsAny().(anthropic.ContentBlockDeltaEvent); ok {
			if t, ok := d.Delta.AsAny().(anthropic.TextDelta); ok && t.Text != "" {
				if err := emit(Chunk{Text: t.Text}); err != nil {
					return Response{}, err
				}
			}
		}
	}
	if err := s.Err(); err != nil {
		return Response{}, anthropicFail(ctx, err)
	}
	return a.result(&acc), nil
}
