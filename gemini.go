package llm

import (
	"context"
	"errors"
	"strings"

	"google.golang.org/genai"
)

type geminiClient struct {
	c     *genai.Client
	model string
}

func newGemini(cfg Config) (Client, error) {
	c, err := genai.NewClient(context.Background(), &genai.ClientConfig{APIKey: cfg.APIKey, Backend: genai.BackendGeminiAPI,
		HTTPClient: cfg.HTTPClient, HTTPOptions: genai.HTTPOptions{BaseURL: cfg.BaseURL}})
	if err != nil {
		return nil, &Error{Kind: ErrBadRequest, Provider: Gemini, Message: "the client could not be set up"}
	}
	return &geminiClient{c: c, model: cfg.Model}, nil
}

func (g *geminiClient) Provider() Provider { return Gemini }

func (g *geminiClient) params(r Request) ([]*genai.Content, *genai.GenerateContentConfig) {
	var contents []*genai.Content
	for _, m := range r.Messages {
		role := genai.Role(genai.RoleUser)
		if m.Role == Assistant {
			role = genai.RoleModel
		}
		contents = append(contents, genai.NewContentFromText(m.Text, role))
	}
	cfg := &genai.GenerateContentConfig{MaxOutputTokens: int32(r.MaxTokens)}
	if r.System != "" {
		cfg.SystemInstruction = genai.NewContentFromText(r.System, genai.RoleUser)
	}
	if r.Temperature != nil {
		cfg.Temperature = genai.Ptr(float32(*r.Temperature))
	}
	return contents, cfg
}

func geminiStop(r genai.FinishReason) string {
	switch r {
	case genai.FinishReasonStop:
		return StopEnd
	case genai.FinishReasonMaxTokens:
		return StopLength
	case genai.FinishReasonSafety, genai.FinishReasonProhibitedContent, genai.FinishReasonBlocklist, genai.FinishReasonRecitation:
		return StopRefused
	}
	return StopOther
}

func geminiFail(ctx context.Context, err error) error {
	var api genai.APIError
	if errors.As(err, &api) {
		return fail(Gemini, ctx, api.Code, err)
	}
	var apiP *genai.APIError
	if errors.As(err, &apiP) && apiP != nil {
		return fail(Gemini, ctx, apiP.Code, err)
	}
	return fail(Gemini, ctx, 0, err)
}

// geminiAdd folds a response, or a streamed piece of one, into the answer so far.
func geminiAdd(out *Response, r *genai.GenerateContentResponse) string {
	var piece strings.Builder
	if len(r.Candidates) > 0 {
		c := r.Candidates[0]
		if c.Content != nil {
			for _, p := range c.Content.Parts {
				if p != nil && !p.Thought {
					piece.WriteString(p.Text)
				}
			}
		}
		if c.FinishReason != "" {
			out.Stop = geminiStop(c.FinishReason)
		}
	} else if r.PromptFeedback != nil && r.PromptFeedback.BlockReason != "" {
		out.Stop = StopRefused
	}
	if r.ModelVersion != "" {
		out.Model = r.ModelVersion
	}
	if u := r.UsageMetadata; u != nil {
		out.Usage = Usage{Input: int(u.PromptTokenCount), Output: int(u.CandidatesTokenCount)}
	}
	out.Text += piece.String()
	return piece.String()
}

func (g *geminiClient) Complete(ctx context.Context, r Request) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	contents, cfg := g.params(r)
	res, err := g.c.Models.GenerateContent(ctx, g.model, contents, cfg)
	if err != nil {
		return Response{}, geminiFail(ctx, err)
	}
	out := Response{Model: g.model, Stop: StopOther}
	geminiAdd(&out, res)
	return out, nil
}

func (g *geminiClient) Stream(ctx context.Context, r Request, emit func(Chunk) error) (Response, error) {
	if err := r.Check(); err != nil {
		return Response{}, err
	}
	contents, cfg := g.params(r)
	out := Response{Model: g.model, Stop: StopOther}
	for res, err := range g.c.Models.GenerateContentStream(ctx, g.model, contents, cfg) {
		if err != nil {
			return Response{}, geminiFail(ctx, err)
		}
		if piece := geminiAdd(&out, res); piece != "" {
			if err := emit(Chunk{Text: piece}); err != nil {
				return Response{}, err
			}
		}
	}
	return out, nil
}
