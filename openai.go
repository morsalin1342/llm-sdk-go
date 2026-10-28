package llm

import (
	"context"
	"errors"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type openaiClient struct {
	c     openai.Client
	model string
}

func newOpenAI(cfg Config) (Client, error) {
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey), option.WithHTTPClient(cfg.HTTPClient), option.WithMaxRetries(0)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &openaiClient{c: openai.NewClient(opts...), model: cfg.Model}, nil
}

func (o *openaiClient) Provider() Provider { return OpenAI }

func (o *openaiClient) params(r Request) openai.ChatCompletionNewParams {
	p := openai.ChatCompletionNewParams{Model: openai.ChatModel(o.model), MaxCompletionTokens: openai.Int(int64(r.MaxTokens))}
	if r.System != "" {
		p.Messages = append(p.Messages, openai.SystemMessage(r.System))
	}
	for _, m := range r.Messages {
		if m.Role == Assistant {
			p.Messages = append(p.Messages, openai.AssistantMessage(m.Text))
		} else {
			p.Messages = append(p.Messages, openai.UserMessage(m.Text))
		}
	}
	if r.Temperature != nil {
		p.Temperature = openai.Float(*r.Temperature)
	}
	return p
}

// openaiStop maps a finish reason of the Chat Completions API (also the shape
// OpenRouter uses) to ours.
func openaiStop(reason string) string {
	switch reason {
	case "stop":
		return StopEnd
	case "length":
		return StopLength
	case "content_filter":
		return StopRefused
	}
	return StopOther
}

func openaiFail(ctx context.Context, err error) error {
	var api *openai.Error
	if errors.As(err, &api) {
		return fail(OpenAI, ctx, api.StatusCode, err)
	}
	return fail(OpenAI, ctx, 0, err)
}

func openaiResult(c *openai.ChatCompletion) Response {
	out := Response{Model: c.Model, Usage: Usage{Input: int(c.Usage.PromptTokens), Output: int(c.Usage.CompletionTokens)}}
	if len(c.Choices) > 0 {
		out.Text, out.Stop = c.Choices[0].Message.Content, openaiStop(c.Choices[0].FinishReason)
		if c.Choices[0].Message.Refusal != "" && out.Text == "" {
			out.Stop = StopRefused
		}
	}
	return out
}

func (o *openaiClient) Complete(ctx context.Context, r Request) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	c, err := o.c.Chat.Completions.New(ctx, o.params(r))
	if err != nil {
		return Response{}, openaiFail(ctx, err)
	}
	return openaiResult(c), nil
}

func (o *openaiClient) Stream(ctx context.Context, r Request, emit func(Chunk) error) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	p := o.params(r)
	p.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
	s := o.c.Chat.Completions.NewStreaming(ctx, p)
	defer s.Close()
	var acc openai.ChatCompletionAccumulator
	for s.Next() {
		chunk := s.Current()
		acc.AddChunk(chunk)
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			if err := emit(Chunk{Text: chunk.Choices[0].Delta.Content}); err != nil {
				return Response{}, err
			}
		}
	}
	if err := s.Err(); err != nil {
		return Response{}, openaiFail(ctx, err)
	}
	return openaiResult(&acc.ChatCompletion), nil
}
