# llm-sdk-go

One Go client for **Anthropic, OpenAI, Google Gemini and OpenRouter**, built on top of each provider's **official Go SDK**.

Write your code once against a small `Client` interface. Switch provider and model by changing a config value.

```go
import llm "github.com/morsalin1342/llm-sdk-go"

c, err := llm.New(llm.Config{
    Provider:   llm.Anthropic,           // or llm.OpenAI, llm.Gemini, llm.OpenRouter
    APIKey:     os.Getenv("API_KEY"),
    Model:      "claude-sonnet-5-5",     // the provider's own model name
    HTTPClient: http.DefaultClient,      // required: you decide how requests leave your process
})
if err != nil { /* ... */ }

res, err := c.Complete(ctx, llm.Request{
    System:    "You are a concise assistant.",
    MaxTokens: 512,
    Messages:  []llm.Message{{Role: llm.User, Text: "Why is my site slow?"}},
})
fmt.Println(res.Text, res.Usage.Input, res.Usage.Output, res.Stop)
```

Streaming:

```go
res, err := c.Stream(ctx, req, func(ch llm.Chunk) error {
    fmt.Print(ch.Text)
    return nil // return an error to stop the stream
})
```

## What you get

| | |
|---|---|
| Providers | Anthropic, OpenAI (Chat Completions), Google Gemini (Gemini API), OpenRouter |
| Calls | `Complete` and `Stream`, text in and text out |
| One answer shape | `Text`, `Stop` (`end`, `length`, `refused`, `other`), `Model`, token `Usage` |
| One error shape | `*llm.Error` with a `Kind`: `auth`, `rate_limit`, `too_large`, `bad_request`, `unavailable`, `canceled`, `internal`. Match with `errors.Is(err, &llm.Error{Kind: llm.ErrRateLimit})` |

## Design choices

- **Official SDKs only.** Each adapter calls the provider's own Go SDK. No hand-written HTTP clients and no compatibility shims.
- **You own the HTTP client.** `Config.HTTPClient` is required and is the only way a request leaves the process, so you can put an egress policy, proxy, tracing or timeouts in it.
- **No hidden retries.** The SDKs' own retries are switched off. You decide when and how to retry; a `rate_limit` or `unavailable` error says it is worth trying again.
- **Keys stay out of errors.** A key is sent to the provider and nowhere else. Errors carry a kind, a status and a short reason, never the key and never the provider's response body.
- **Same behaviour, tested the same way.** One test suite runs every adapter against a local fake of that provider's real wire format. No keys, no network.

## Not yet

Tool calls, images, structured output and embeddings. Open an issue if you need one.

## SDKs this builds on

| Provider | SDK | Version |
|---|---|---|
| Anthropic | [anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) | v1.81.0 |
| OpenAI | [openai-go](https://github.com/openai/openai-go) | v3.78.0 |
| Google Gemini | [google.golang.org/genai](https://pkg.go.dev/google.golang.org/genai) | v1.75.0 |
| OpenRouter | [go-sdk](https://github.com/OpenRouterTeam/go-sdk) (beta) | v0.9.54 |

Requires Go 1.26 or newer.

## License

Apache-2.0. See [LICENSE](LICENSE).
