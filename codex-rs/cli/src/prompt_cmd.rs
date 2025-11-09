use anyhow::Context;
use clap::Parser;
use clap::ValueHint;
use codex_app_server_protocol::AuthMode;
use codex_common::CliConfigOverrides;
use codex_common::model_presets::builtin_model_presets;
use codex_core::AuthManager;
use codex_core::ModelClient;
use codex_core::OtelEventManager;
use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_core::ResponseStream;
use codex_core::auth::enforce_login_restrictions;
use codex_core::config::Config;
use codex_core::config::ConfigOverrides;
use codex_core::terminal;
use codex_protocol::ConversationId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::TokenUsage;
use futures::StreamExt;
use std::io::IsTerminal;
use std::io::Read;
use std::io::Write;
use std::sync::Arc;

/// Run a single prompt directly against the configured model.
#[derive(Debug, Parser)]
pub struct PromptCli {
    #[clap(flatten)]
    pub config_overrides: CliConfigOverrides,

    /// Model to use for this request.
    #[arg(long, short = 'm', value_name = "MODEL")]
    pub model: Option<String>,

    /// Override the developer/system instructions for this request.
    #[arg(long = "system", short = 's', value_name = "SYSTEM_PROMPT")]
    pub system_prompt: Option<String>,

    /// List models that can be used with `codex prompt`.
    #[arg(long = "models", conflicts_with = "prompt", default_value_t = false)]
    pub list_models: bool,

    /// Prompt to send to the model. Use `-` to read from stdin.
    #[arg(value_name = "PROMPT", value_hint = ValueHint::Other)]
    pub prompt: Option<String>,

    /// Print the outgoing JSON request and incoming SSE payloads.
    #[arg(long = "debug", default_value_t = false)]
    pub debug: bool,
}

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant. Respond directly to the user request without running tools or shell commands.";

pub async fn run_prompt_command(cli: PromptCli) -> anyhow::Result<()> {
    let prompt_text = if cli.list_models {
        None
    } else {
        Some(read_prompt(cli.prompt.clone())?)
    };

    let system_prompt = cli
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    let config = Arc::new(load_config(&cli).await?);
    let auth_manager = AuthManager::shared(
        config.codex_home.clone(),
        true,
        config.cli_auth_credentials_store_mode,
    );

    if cli.list_models {
        print_models(auth_manager.auth().map(|auth| auth.mode));
        return Ok(());
    }

    if let Err(err) = enforce_login_restrictions(&config).await {
        eprintln!("{err}");
        std::process::exit(1);
    }

    let prompt_text = prompt_text.ok_or_else(|| anyhow::anyhow!("prompt is required"))?;
    run_prompt(prompt_text, system_prompt, config, auth_manager, cli.debug).await
}

async fn load_config(cli: &PromptCli) -> anyhow::Result<Config> {
    let overrides = ConfigOverrides {
        model: cli.model.clone(),
        review_model: None,
        cwd: None,
        approval_policy: None,
        sandbox_mode: None,
        model_provider: None,
        config_profile: None,
        codex_linux_sandbox_exe: None,
        base_instructions: None,
        developer_instructions: None,
        compact_prompt: None,
        include_apply_patch_tool: Some(false),
        show_raw_agent_reasoning: None,
        tools_web_search_request: Some(false),
        experimental_sandbox_command_assessment: Some(false),
        additional_writable_roots: Vec::new(),
    };

    let cli_overrides = cli
        .config_overrides
        .parse_overrides()
        .map_err(anyhow::Error::msg)?;

    Config::load_with_cli_overrides(cli_overrides, overrides)
        .await
        .map_err(anyhow::Error::from)
}

fn read_prompt(prompt: Option<String>) -> anyhow::Result<String> {
    match prompt {
        Some(p) if p != "-" => Ok(p),
        other => {
            let force_stdin = matches!(other.as_deref(), Some("-"));
            if std::io::stdin().is_terminal() && !force_stdin {
                anyhow::bail!("No prompt provided. Pass one as an argument or pipe it via stdin.");
            }
            if !force_stdin {
                eprintln!("Reading prompt from stdin...");
            }
            let mut buffer = String::new();
            std::io::stdin()
                .read_to_string(&mut buffer)
                .context("Failed to read prompt from stdin")?;
            if buffer.trim().is_empty() {
                anyhow::bail!("No prompt provided via stdin.");
            }
            Ok(buffer)
        }
    }
}

fn print_models(auth_mode: Option<AuthMode>) {
    let presets = builtin_model_presets(auth_mode);
    if presets.is_empty() {
        println!("No models are currently available.");
        return;
    }

    println!("Available models:");
    for preset in presets {
        let default_marker = if preset.is_default { " (default)" } else { "" };
        println!(
            "  {}{} - {}",
            preset.model, default_marker, preset.description
        );
        println!(
            "    Default reasoning effort: {}",
            preset.default_reasoning_effort
        );
        if !preset.supported_reasoning_efforts.is_empty() {
            println!("    Supported reasoning efforts:");
            for option in preset.supported_reasoning_efforts {
                println!("      - {}: {}", option.effort, option.description);
            }
        }
    }
}

async fn run_prompt(
    prompt_text: String,
    system_prompt: String,
    config: Arc<Config>,
    auth_manager: Arc<AuthManager>,
    debug_http: bool,
) -> anyhow::Result<()> {
    let auth_snapshot = auth_manager.auth();
    let provider = config.model_provider.clone();
    let conversation_id = ConversationId::new();
    let otel_event_manager = OtelEventManager::new(
        conversation_id,
        config.model.as_str(),
        config.model_family.slug.as_str(),
        auth_snapshot
            .as_ref()
            .and_then(codex_core::CodexAuth::get_account_id),
        auth_snapshot
            .as_ref()
            .and_then(codex_core::CodexAuth::get_account_email),
        auth_snapshot.as_ref().map(|auth| auth.mode),
        config.otel.log_user_prompt,
        terminal::user_agent(),
    );

    let mut prompt = Prompt::default();
    prompt.input = build_prompt_inputs(&system_prompt, &prompt_text);
    prompt.base_instructions_override = config.base_instructions.clone();

    let mut stream = ModelClient::new(
        Arc::clone(&config),
        Some(auth_manager),
        otel_event_manager,
        provider,
        config.model_reasoning_effort,
        config.model_reasoning_summary,
        conversation_id,
        SessionSource::Cli,
        debug_http,
    )
    .stream(&prompt)
    .await?;

    consume_stream(&mut stream).await
}

fn build_prompt_inputs(system_prompt: &str, prompt_text: &str) -> Vec<ResponseItem> {
    vec![
        ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: system_prompt.to_string(),
            }],
        },
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: prompt_text.to_string(),
            }],
        },
    ]
}

async fn consume_stream(stream: &mut ResponseStream) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let mut printed_response = false;
    let mut reasoning_summary_line = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::Created => {}
            ResponseEvent::OutputTextDelta(delta) => {
                stdout.write_all(delta.as_bytes())?;
                stdout.flush()?;
                printed_response = true;
            }
            ResponseEvent::OutputItemAdded(item) | ResponseEvent::OutputItemDone(item) => {
                if let Some(text) = assistant_text(&item)
                    && !printed_response
                {
                    stdout.write_all(text.as_bytes())?;
                    stdout.flush()?;
                    printed_response = true;
                }
            }
            ResponseEvent::ReasoningSummaryDelta(text) => {
                reasoning_summary_line.push_str(&text);
                eprint!("\r(reasoning summary) {reasoning_summary_line}");
                stderr.flush()?;
            }
            ResponseEvent::ReasoningContentDelta(text) => {
                eprintln!("(reasoning detail) {text}");
            }
            ResponseEvent::ReasoningSummaryPartAdded => {
                if !reasoning_summary_line.is_empty() {
                    eprintln!();
                    reasoning_summary_line.clear();
                }
            }
            ResponseEvent::RateLimits(snapshot) => {
                eprintln!("Rate limits: {snapshot:?}");
            }
            ResponseEvent::Completed { token_usage, .. } => {
                if !reasoning_summary_line.is_empty() {
                    eprintln!();
                    reasoning_summary_line.clear();
                }
                if printed_response {
                    stdout.write_all(b"\n")?;
                    stdout.flush()?;
                    printed_response = false;
                }
                if let Some(usage) = token_usage {
                    print_token_usage(&usage);
                }
            }
        }
    }

    if printed_response {
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    if !reasoning_summary_line.is_empty() {
        eprintln!();
    }
    Ok(())
}

fn assistant_text(item: &ResponseItem) -> Option<String> {
    if let ResponseItem::Message { role, content, .. } = item
        && role == "assistant"
    {
        let mut text = String::new();
        for chunk in content {
            match chunk {
                ContentItem::InputText { text: value }
                | ContentItem::OutputText { text: value } => text.push_str(value),
                ContentItem::InputImage { .. } => {}
            }
        }
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

fn print_token_usage(usage: &TokenUsage) {
    eprintln!(
        "Token usage: total={} input={} cached_input={} output={} reasoning_output={}",
        usage.total_tokens,
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_output_tokens
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_text_handles_basic_message() {
        let item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![
                ContentItem::OutputText {
                    text: "Hello".to_string(),
                },
                ContentItem::OutputText {
                    text: " world".to_string(),
                },
            ],
        };
        assert_eq!(assistant_text(&item), Some("Hello world".to_string()));
    }
}
