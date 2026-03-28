#[allow(clippy::wildcard_imports)]
use super::*;
use crate::config::traits::ChannelConfig;
// ?? Channels ????????????????????????????????????????????????????????
struct ConfigWrapper<T: ChannelConfig>(std::marker::PhantomData<T>);

impl<T: ChannelConfig> ConfigWrapper<T> {
    fn new(_: Option<&T>) -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: ChannelConfig> crate::config::traits::ConfigHandle for ConfigWrapper<T> {
    fn name(&self) -> &'static str {
        T::name()
    }
    fn desc(&self) -> &'static str {
        T::desc()
    }
}

/// Top-level channel configurations (`[channels_config]` section).
///
/// Each channel sub-section (e.g. `telegram`, `discord`) is optional;
/// setting it to `Some(...)` enables that channel.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelsConfig {
    /// Enable the CLI interactive channel. Default: `true`.
    #[serde(default = "default_true")]
    pub cli: bool,
    /// Telegram bot channel configuration.
    pub telegram: Option<TelegramConfig>,
    /// Discord bot channel configuration.
    pub discord: Option<DiscordConfig>,
    /// Slack bot channel configuration.
    pub slack: Option<SlackConfig>,
    /// Mattermost bot channel configuration.
    pub mattermost: Option<MattermostConfig>,
    /// Webhook channel configuration.
    pub webhook: Option<WebhookConfig>,
    /// iMessage channel configuration (macOS only).
    pub imessage: Option<IMessageConfig>,
    /// Matrix channel configuration.
    pub matrix: Option<MatrixConfig>,
    /// Signal channel configuration.
    pub signal: Option<SignalConfig>,
    /// WhatsApp channel configuration (Cloud API or Web mode).
    pub whatsapp: Option<WhatsAppConfig>,
    /// Linq Partner API channel configuration.
    pub linq: Option<LinqConfig>,
    /// WATI WhatsApp Business API channel configuration.
    pub wati: Option<WatiConfig>,
    /// Nextcloud Talk bot channel configuration.
    pub nextcloud_talk: Option<NextcloudTalkConfig>,
    /// Email channel configuration.
    pub email: Option<crate::channels::email_channel::EmailConfig>,
    /// IRC channel configuration.
    pub irc: Option<IrcConfig>,
    /// Lark channel configuration.
    pub lark: Option<LarkConfig>,
    /// Feishu channel configuration.
    pub feishu: Option<FeishuConfig>,
    /// DingTalk channel configuration.
    pub dingtalk: Option<DingTalkConfig>,
    /// WeCom (WeChat Enterprise) Bot Webhook channel configuration.
    pub wecom: Option<WeComConfig>,
    /// QQ Official Bot channel configuration.
    pub qq: Option<QQConfig>,
    /// X/Twitter channel configuration.
    pub twitter: Option<TwitterConfig>,
    /// Mochat customer service channel configuration.
    pub mochat: Option<MochatConfig>,
    #[cfg(feature = "channel-nostr")]
    pub nostr: Option<NostrConfig>,
    /// ClawdTalk voice channel configuration.
    pub clawdtalk: Option<crate::channels::ClawdTalkConfig>,
    /// Reddit channel configuration (OAuth2 bot).
    pub reddit: Option<RedditConfig>,
    /// Bluesky channel configuration (AT Protocol).
    pub bluesky: Option<BlueskyConfig>,
    /// Base timeout in seconds for processing a single channel message (LLM + tools).
    /// Runtime uses this as a per-turn budget that scales with tool-loop depth
    /// (up to 4x, capped) so one slow/retried model call does not consume the
    /// entire conversation budget.
    /// Default: 300s for on-device LLMs (Ollama) which are slower than cloud APIs.
    #[serde(default = "default_channel_message_timeout_secs")]
    pub message_timeout_secs: u64,
    /// Whether to add acknowledgement reactions (👀 on receipt, ✅/⚠️ on
    /// completion) to incoming channel messages. Default: `true`.
    #[serde(default = "default_true")]
    pub ack_reactions: bool,
    /// Whether to send tool-call notification messages (e.g. `🔧 web_search_tool: …`)
    /// to channel users. When `false`, tool calls are still logged server-side but
    /// not forwarded as individual channel messages. Default: `false`.
    #[serde(default = "default_false")]
    pub show_tool_calls: bool,
    /// Persist channel conversation history to JSONL files so sessions survive
    /// daemon restarts. Files are stored in `{workspace}/sessions/`. Default: `true`.
    #[serde(default = "default_true")]
    pub session_persistence: bool,
    /// Session persistence backend: `"jsonl"` (legacy) or `"sqlite"` (new default).
    /// SQLite provides FTS5 search, metadata tracking, and TTL cleanup.
    #[serde(default = "default_session_backend")]
    pub session_backend: String,
    /// Auto-archive stale sessions older than this many hours. `0` disables. Default: `0`.
    #[serde(default)]
    pub session_ttl_hours: u32,
}

impl ChannelsConfig {
    /// get channels' metadata and `.is_some()`, except webhook
    #[rustfmt::skip]
    pub fn channels_except_webhook(
        &self,
    ) -> Vec<(Box<dyn crate::config::traits::ConfigHandle>, bool)> {
        vec![
            (
                Box::new(ConfigWrapper::new(self.telegram.as_ref())),
                self.telegram.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.discord.as_ref())),
                self.discord.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.slack.as_ref())),
                self.slack.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.mattermost.as_ref())),
                self.mattermost.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.imessage.as_ref())),
                self.imessage.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.matrix.as_ref())),
                self.matrix.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.signal.as_ref())),
                self.signal.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.whatsapp.as_ref())),
                self.whatsapp.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.linq.as_ref())),
                self.linq.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.wati.as_ref())),
                self.wati.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.nextcloud_talk.as_ref())),
                self.nextcloud_talk.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.email.as_ref())),
                self.email.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.irc.as_ref())),
                self.irc.is_some()
            ),
            (
                Box::new(ConfigWrapper::new(self.lark.as_ref())),
                self.lark.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.feishu.as_ref())),
                self.feishu.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.dingtalk.as_ref())),
                self.dingtalk.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.wecom.as_ref())),
                self.wecom.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.qq.as_ref())),
                self.qq.is_some()
            ),
            #[cfg(feature = "channel-nostr")]
            (
                Box::new(ConfigWrapper::new(self.nostr.as_ref())),
                self.nostr.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.clawdtalk.as_ref())),
                self.clawdtalk.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.reddit.as_ref())),
                self.reddit.is_some(),
            ),
            (
                Box::new(ConfigWrapper::new(self.bluesky.as_ref())),
                self.bluesky.is_some(),
            ),
        ]
    }

    pub fn channels(&self) -> Vec<(Box<dyn crate::config::traits::ConfigHandle>, bool)> {
        let mut ret = self.channels_except_webhook();
        ret.push((
            Box::new(ConfigWrapper::new(self.webhook.as_ref())),
            self.webhook.is_some(),
        ));
        ret
    }
}

pub(crate) fn default_channel_message_timeout_secs() -> u64 {
    300
}

pub(crate) fn default_session_backend() -> String {
    "sqlite".into()
}

impl Default for ChannelsConfig {
    fn default() -> Self {
        Self {
            cli: true,
            telegram: None,
            discord: None,
            slack: None,
            mattermost: None,
            webhook: None,
            imessage: None,
            matrix: None,
            signal: None,
            whatsapp: None,
            linq: None,
            wati: None,
            nextcloud_talk: None,
            email: None,
            irc: None,
            lark: None,
            feishu: None,
            dingtalk: None,
            wecom: None,
            qq: None,
            twitter: None,
            mochat: None,
            #[cfg(feature = "channel-nostr")]
            nostr: None,
            clawdtalk: None,
            reddit: None,
            bluesky: None,
            message_timeout_secs: default_channel_message_timeout_secs(),
            ack_reactions: true,
            show_tool_calls: false,
            session_persistence: true,
            session_backend: default_session_backend(),
            session_ttl_hours: 0,
        }
    }
}

/// Streaming mode for channels that support progressive message updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamMode {
    /// No streaming -- send the complete response as a single message (default).
    #[default]
    Off,
    /// Update a draft message with every flush interval.
    Partial,
}

pub(crate) fn default_draft_update_interval_ms() -> u64 {
    1000
}

/// Telegram bot channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TelegramConfig {
    /// Telegram Bot API token (from @BotFather).
    pub bot_token: String,
    /// Allowed Telegram user IDs or usernames. Empty = deny all.
    pub allowed_users: Vec<String>,
    /// Streaming mode for progressive response delivery via message edits.
    #[serde(default)]
    pub stream_mode: StreamMode,
    /// Minimum interval (ms) between draft message edits to avoid rate limits.
    #[serde(default = "default_draft_update_interval_ms")]
    pub draft_update_interval_ms: u64,
    /// When true, a newer Telegram message from the same sender in the same chat
    /// cancels the in-flight request and starts a fresh response with preserved history.
    #[serde(default)]
    pub interrupt_on_new_message: bool,
    /// When true, only respond to messages that @-mention the bot in groups.
    /// Direct messages are always processed.
    #[serde(default)]
    pub mention_only: bool,
    /// Override for the top-level `ack_reactions` setting. When `None`, the
    /// channel falls back to `[channels_config].ack_reactions`. When set
    /// explicitly, it takes precedence.
    #[serde(default)]
    pub ack_reactions: Option<bool>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for TelegramConfig {
    fn name() -> &'static str {
        "Telegram"
    }
    fn desc() -> &'static str {
        "connect your bot"
    }
}

/// Discord bot channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DiscordConfig {
    /// Discord bot token (from Discord Developer Portal).
    pub bot_token: String,
    /// Optional guild (server) ID to restrict the bot to a single guild.
    pub guild_id: Option<String>,
    /// Allowed Discord user IDs. Empty = deny all.
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// When true, process messages from other bots (not just humans).
    /// The bot still ignores its own messages to prevent feedback loops.
    #[serde(default)]
    pub listen_to_bots: bool,
    /// When true, a newer Discord message from the same sender in the same channel
    /// cancels the in-flight request and starts a fresh response with preserved history.
    #[serde(default)]
    pub interrupt_on_new_message: bool,
    /// When true, only respond to messages that @-mention the bot.
    /// Other messages in the guild are silently ignored.
    #[serde(default)]
    pub mention_only: bool,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for DiscordConfig {
    fn name() -> &'static str {
        "Discord"
    }
    fn desc() -> &'static str {
        "connect your bot"
    }
}

/// Slack bot channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SlackConfig {
    /// Slack bot OAuth token (xoxb-...).
    pub bot_token: String,
    /// Slack app-level token for Socket Mode (xapp-...).
    pub app_token: Option<String>,
    /// Optional channel ID to restrict the bot to a single channel.
    /// Omit (or set `"*"`) to listen across all accessible channels.
    pub channel_id: Option<String>,
    /// Allowed Slack user IDs. Empty = deny all.
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// When true, a newer Slack message from the same sender in the same channel
    /// cancels the in-flight request and starts a fresh response with preserved history.
    #[serde(default)]
    pub interrupt_on_new_message: bool,
    /// When true (default), replies stay in the originating Slack thread.
    /// When false, replies go to the channel root instead.
    #[serde(default)]
    pub thread_replies: Option<bool>,
    /// When true, only respond to messages that @-mention the bot in groups.
    /// Direct messages remain allowed.
    #[serde(default)]
    pub mention_only: bool,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for SlackConfig {
    fn name() -> &'static str {
        "Slack"
    }
    fn desc() -> &'static str {
        "connect your bot"
    }
}

/// Mattermost bot channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MattermostConfig {
    /// Mattermost server URL (e.g. `"https://mattermost.example.com"`).
    pub url: String,
    /// Mattermost bot access token.
    pub bot_token: String,
    /// Optional channel ID to restrict the bot to a single channel.
    pub channel_id: Option<String>,
    /// Allowed Mattermost user IDs. Empty = deny all.
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// When true (default), replies thread on the original post.
    /// When false, replies go to the channel root.
    #[serde(default)]
    pub thread_replies: Option<bool>,
    /// When true, only respond to messages that @-mention the bot.
    /// Other messages in the channel are silently ignored.
    #[serde(default)]
    pub mention_only: Option<bool>,
    /// When true, a newer Mattermost message from the same sender in the same channel
    /// cancels the in-flight request and starts a fresh response with preserved history.
    #[serde(default)]
    pub interrupt_on_new_message: bool,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for MattermostConfig {
    fn name() -> &'static str {
        "Mattermost"
    }
    fn desc() -> &'static str {
        "connect to your bot"
    }
}

/// Webhook channel configuration.
///
/// Receives messages via HTTP POST and sends replies to a configurable outbound URL.
/// This is the "universal adapter" for any system that supports webhooks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebhookConfig {
    /// Port to listen on for incoming webhooks.
    pub port: u16,
    /// URL path to listen on (default: `/webhook`).
    #[serde(default)]
    pub listen_path: Option<String>,
    /// URL to POST/PUT outbound messages to.
    #[serde(default)]
    pub send_url: Option<String>,
    /// HTTP method for outbound messages (`POST` or `PUT`). Default: `POST`.
    #[serde(default)]
    pub send_method: Option<String>,
    /// Optional `Authorization` header value for outbound requests.
    #[serde(default)]
    pub auth_header: Option<String>,
    /// Optional shared secret for webhook signature verification (HMAC-SHA256).
    pub secret: Option<String>,
}

impl ChannelConfig for WebhookConfig {
    fn name() -> &'static str {
        "Webhook"
    }
    fn desc() -> &'static str {
        "HTTP endpoint"
    }
}

/// iMessage channel configuration (macOS only).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IMessageConfig {
    /// Allowed iMessage contacts (phone numbers or email addresses). Empty = deny all.
    pub allowed_contacts: Vec<String>,
}

impl ChannelConfig for IMessageConfig {
    fn name() -> &'static str {
        "iMessage"
    }
    fn desc() -> &'static str {
        "macOS only"
    }
}

/// Matrix channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MatrixConfig {
    /// Matrix homeserver URL (e.g. `"https://matrix.org"`).
    pub homeserver: String,
    /// Matrix access token for the bot account.
    pub access_token: String,
    /// Optional Matrix user ID (e.g. `"@bot:matrix.org"`).
    #[serde(default)]
    pub user_id: Option<String>,
    /// Optional Matrix device ID.
    #[serde(default)]
    pub device_id: Option<String>,
    /// Matrix room ID to listen in (e.g. `"!abc123:matrix.org"`).
    pub room_id: String,
    /// Allowed Matrix user IDs. Empty = deny all.
    pub allowed_users: Vec<String>,
    /// Whether to interrupt an in-flight agent response when a new message arrives.
    #[serde(default)]
    pub interrupt_on_new_message: bool,
}

impl ChannelConfig for MatrixConfig {
    fn name() -> &'static str {
        "Matrix"
    }
    fn desc() -> &'static str {
        "self-hosted chat"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SignalConfig {
    /// Base URL for the signal-cli HTTP daemon (e.g. "http://127.0.0.1:8686").
    pub http_url: String,
    /// E.164 phone number of the signal-cli account (e.g. "+1234567890").
    pub account: String,
    /// Optional group ID to filter messages.
    /// - `None` or omitted: accept all messages (DMs and groups)
    /// - `"dm"`: only accept direct messages
    /// - Specific group ID: only accept messages from that group
    #[serde(default)]
    pub group_id: Option<String>,
    /// Allowed sender phone numbers (E.164) or "*" for all.
    #[serde(default)]
    pub allowed_from: Vec<String>,
    /// Skip messages that are attachment-only (no text body).
    #[serde(default)]
    pub ignore_attachments: bool,
    /// Skip incoming story messages.
    #[serde(default)]
    pub ignore_stories: bool,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for SignalConfig {
    fn name() -> &'static str {
        "Signal"
    }
    fn desc() -> &'static str {
        "An open-source, encrypted messaging service"
    }
}

/// WhatsApp Web usage mode.
///
/// `Personal` treats the account as a personal phone — the bot only responds to
/// incoming messages that pass the DM/group/self-chat policy filters.
/// `Business` (default) responds to all incoming messages, subject only to the
/// `allowed_numbers` allowlist.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WhatsAppWebMode {
    /// Respond to all messages passing the allowlist (default).
    #[default]
    Business,
    /// Apply per-chat-type policies (dm_policy, group_policy, self_chat_mode).
    Personal,
}

/// Policy for a particular WhatsApp chat type (DMs or groups) when
/// `mode = "personal"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WhatsAppChatPolicy {
    /// Only respond to senders on the `allowed_numbers` list (default).
    #[default]
    Allowlist,
    /// Ignore all messages in this chat type.
    Ignore,
    /// Respond to every message regardless of allowlist.
    All,
}

/// WhatsApp channel configuration (Cloud API or Web mode).
///
/// Set `phone_number_id` for Cloud API mode, or `session_path` for Web mode.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WhatsAppConfig {
    /// Access token from Meta Business Suite (Cloud API mode)
    #[serde(default)]
    pub access_token: Option<String>,
    /// Phone number ID from Meta Business API (Cloud API mode)
    #[serde(default)]
    pub phone_number_id: Option<String>,
    /// Webhook verify token (you define this, Meta sends it back for verification)
    /// Only used in Cloud API mode
    #[serde(default)]
    pub verify_token: Option<String>,
    /// App secret from Meta Business Suite (for webhook signature verification)
    /// Can also be set via `rain_WHATSAPP_APP_SECRET` environment variable
    /// Only used in Cloud API mode
    #[serde(default)]
    pub app_secret: Option<String>,
    /// Session database path for WhatsApp Web client (Web mode)
    /// When set, enables native WhatsApp Web mode with wa-rs
    #[serde(default)]
    pub session_path: Option<String>,
    /// Phone number for pair code linking (Web mode, optional)
    /// Format: country code + number (e.g., "15551234567")
    /// If not set, QR code pairing will be used
    #[serde(default)]
    pub pair_phone: Option<String>,
    /// Custom pair code for linking (Web mode, optional)
    /// Leave empty to let WhatsApp generate one
    #[serde(default)]
    pub pair_code: Option<String>,
    /// Allowed phone numbers (E.164 format: +1234567890) or "*" for all
    #[serde(default)]
    pub allowed_numbers: Vec<String>,
    /// Usage mode for WhatsApp Web: "business" (default) or "personal".
    /// In personal mode the bot applies dm_policy, group_policy, and
    /// self_chat_mode to decide which chats to respond in.
    #[serde(default)]
    pub mode: WhatsAppWebMode,
    /// Policy for direct messages when mode = "personal".
    /// "allowlist" (default) | "ignore" | "all".
    #[serde(default)]
    pub dm_policy: WhatsAppChatPolicy,
    /// Policy for group chats when mode = "personal".
    /// "allowlist" (default) | "ignore" | "all".
    #[serde(default)]
    pub group_policy: WhatsAppChatPolicy,
    /// When true and mode = "personal", always respond to messages in the
    /// user's own self-chat (Notes to Self). Defaults to false.
    #[serde(default)]
    pub self_chat_mode: bool,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for WhatsAppConfig {
    fn name() -> &'static str {
        "WhatsApp"
    }
    fn desc() -> &'static str {
        "Business Cloud API"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LinqConfig {
    /// Linq Partner API token (Bearer auth)
    pub api_token: String,
    /// Phone number to send from (E.164 format)
    pub from_phone: String,
    /// Webhook signing secret for signature verification
    #[serde(default)]
    pub signing_secret: Option<String>,
    /// Allowed sender handles (phone numbers) or "*" for all
    #[serde(default)]
    pub allowed_senders: Vec<String>,
}

impl ChannelConfig for LinqConfig {
    fn name() -> &'static str {
        "Linq"
    }
    fn desc() -> &'static str {
        "iMessage/RCS/SMS via Linq API"
    }
}

/// WATI WhatsApp Business API channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WatiConfig {
    /// WATI API token (Bearer auth).
    pub api_token: String,
    /// WATI API base URL (default: https://live-mt-server.wati.io).
    #[serde(default = "default_wati_api_url")]
    pub api_url: String,
    /// Tenant ID for multi-channel setups (optional).
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Allowed phone numbers (E.164 format) or "*" for all.
    #[serde(default)]
    pub allowed_numbers: Vec<String>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

fn default_wati_api_url() -> String {
    "https://live-mt-server.wati.io".to_string()
}

impl ChannelConfig for WatiConfig {
    fn name() -> &'static str {
        "WATI"
    }
    fn desc() -> &'static str {
        "WhatsApp via WATI Business API"
    }
}

/// Nextcloud Talk bot configuration (webhook receive + OCS send API).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NextcloudTalkConfig {
    /// Nextcloud base URL (e.g. "https://cloud.example.com").
    pub base_url: String,
    /// Bot app token used for OCS API bearer auth.
    pub app_token: String,
    /// Shared secret for webhook signature verification.
    ///
    /// Can also be set via `rain_NEXTCLOUD_TALK_WEBHOOK_SECRET`.
    #[serde(default)]
    pub webhook_secret: Option<String>,
    /// Allowed Nextcloud actor IDs (`[]` = deny all, `"*"` = allow all).
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for NextcloudTalkConfig {
    fn name() -> &'static str {
        "NextCloud Talk"
    }
    fn desc() -> &'static str {
        "NextCloud Talk platform"
    }
}

impl WhatsAppConfig {
    /// Detect which backend to use based on config fields.
    /// Returns "cloud" if phone_number_id is set, "web" if session_path is set.
    pub fn backend_type(&self) -> &'static str {
        if self.phone_number_id.is_some() {
            "cloud"
        } else if self.session_path.is_some() {
            "web"
        } else {
            // Default to Cloud API for backward compatibility
            "cloud"
        }
    }

    /// Check if this is a valid Cloud API config
    pub fn is_cloud_config(&self) -> bool {
        self.phone_number_id.is_some() && self.access_token.is_some() && self.verify_token.is_some()
    }

    /// Check if this is a valid Web config
    pub fn is_web_config(&self) -> bool {
        self.session_path.is_some()
    }

    /// Returns true when both Cloud and Web selectors are present.
    ///
    /// Runtime currently prefers Cloud mode in this case for backward compatibility.
    pub fn is_ambiguous_config(&self) -> bool {
        self.phone_number_id.is_some() && self.session_path.is_some()
    }
}

/// IRC channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IrcConfig {
    /// IRC server hostname
    pub server: String,
    /// IRC server port (default: 6697 for TLS)
    #[serde(default = "default_irc_port")]
    pub port: u16,
    /// Bot nickname
    pub nickname: String,
    /// Username (defaults to nickname if not set)
    pub username: Option<String>,
    /// Channels to join on connect
    #[serde(default)]
    pub channels: Vec<String>,
    /// Allowed nicknames (case-insensitive) or "*" for all
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Server password (for bouncers like ZNC)
    pub server_password: Option<String>,
    /// NickServ IDENTIFY password
    pub nickserv_password: Option<String>,
    /// SASL PLAIN password (IRCv3)
    pub sasl_password: Option<String>,
    /// Verify TLS certificate (default: true)
    pub verify_tls: Option<bool>,
}

impl ChannelConfig for IrcConfig {
    fn name() -> &'static str {
        "IRC"
    }
    fn desc() -> &'static str {
        "IRC over TLS"
    }
}

fn default_irc_port() -> u16 {
    6697
}

/// How R.A.I.N. receives events from Feishu / Lark.
///
/// - `websocket` (default) — persistent WSS long-connection; no public URL required.
/// - `webhook`             — HTTP callback server; requires a public HTTPS endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LarkReceiveMode {
    #[default]
    Websocket,
    Webhook,
}

/// Lark/Feishu configuration for messaging integration.
/// Lark is the international version; Feishu is the Chinese version.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LarkConfig {
    /// App ID from Lark/Feishu developer console
    pub app_id: String,
    /// App Secret from Lark/Feishu developer console
    pub app_secret: String,
    /// Encrypt key for webhook message decryption (optional)
    #[serde(default)]
    pub encrypt_key: Option<String>,
    /// Verification token for webhook validation (optional)
    #[serde(default)]
    pub verification_token: Option<String>,
    /// Allowed user IDs or union IDs (empty = deny all, "*" = allow all)
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// When true, only respond to messages that @-mention the bot in groups.
    /// Direct messages are always processed.
    #[serde(default)]
    pub mention_only: bool,
    /// Whether to use the Feishu (Chinese) endpoint instead of Lark (International)
    #[serde(default)]
    pub use_feishu: bool,
    /// Event receive mode: "websocket" (default) or "webhook"
    #[serde(default)]
    pub receive_mode: LarkReceiveMode,
    /// HTTP port for webhook mode only. Must be set when receive_mode = "webhook".
    /// Not required (and ignored) for websocket mode.
    #[serde(default)]
    pub port: Option<u16>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for LarkConfig {
    fn name() -> &'static str {
        "Lark"
    }
    fn desc() -> &'static str {
        "Lark Bot"
    }
}

/// Feishu configuration for messaging integration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FeishuConfig {
    /// App ID from Feishu developer console
    pub app_id: String,
    /// App Secret from Feishu developer console
    pub app_secret: String,
    /// Encrypt key for webhook message decryption (optional)
    #[serde(default)]
    pub encrypt_key: Option<String>,
    /// Verification token for webhook validation (optional)
    #[serde(default)]
    pub verification_token: Option<String>,
    /// Allowed user IDs or union IDs (empty = deny all, "*" = allow all)
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Event receive mode: "websocket" (default) or "webhook"
    #[serde(default)]
    pub receive_mode: LarkReceiveMode,
    /// HTTP port for webhook mode only. Must be set when receive_mode = "webhook".
    /// Not required (and ignored) for websocket mode.
    #[serde(default)]
    pub port: Option<u16>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for FeishuConfig {
    fn name() -> &'static str {
        "Feishu"
    }
    fn desc() -> &'static str {
        "Feishu Bot"
    }
}

// ── Security Config ─────────────────────────────────────────────────

/// DingTalk configuration for Stream Mode messaging
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DingTalkConfig {
    /// Client ID (AppKey) from DingTalk developer console
    pub client_id: String,
    /// Client Secret (AppSecret) from DingTalk developer console
    pub client_secret: String,
    /// Allowed user IDs (staff IDs). Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for DingTalkConfig {
    fn name() -> &'static str {
        "DingTalk"
    }
    fn desc() -> &'static str {
        "DingTalk Stream Mode"
    }
}

/// WeCom (WeChat Enterprise) Bot Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeComConfig {
    /// Webhook key from WeCom Bot configuration
    pub webhook_key: String,
    /// Allowed user IDs. Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_users: Vec<String>,
}

impl ChannelConfig for WeComConfig {
    fn name() -> &'static str {
        "WeCom"
    }
    fn desc() -> &'static str {
        "WeCom Bot Webhook"
    }
}

/// QQ Official Bot configuration (Tencent QQ Bot SDK)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QQConfig {
    /// App ID from QQ Bot developer console
    pub app_id: String,
    /// App Secret from QQ Bot developer console
    pub app_secret: String,
    /// Allowed user IDs. Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Per-channel proxy URL (http, https, socks5, socks5h).
    /// Overrides the global `[proxy]` setting for this channel only.
    #[serde(default)]
    pub proxy_url: Option<String>,
}

impl ChannelConfig for QQConfig {
    fn name() -> &'static str {
        "QQ Official"
    }
    fn desc() -> &'static str {
        "Tencent QQ Bot"
    }
}

/// X/Twitter channel configuration (Twitter API v2)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TwitterConfig {
    /// Twitter API v2 Bearer Token (OAuth 2.0)
    pub bearer_token: String,
    /// Allowed usernames or user IDs. Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_users: Vec<String>,
}

impl ChannelConfig for TwitterConfig {
    fn name() -> &'static str {
        "X/Twitter"
    }
    fn desc() -> &'static str {
        "X/Twitter Bot via API v2"
    }
}

/// Mochat channel configuration (Mochat customer service API)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MochatConfig {
    /// Mochat API base URL
    pub api_url: String,
    /// Mochat API token
    pub api_token: String,
    /// Allowed user IDs. Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Poll interval in seconds for new messages. Default: 5
    #[serde(default = "default_mochat_poll_interval")]
    pub poll_interval_secs: u64,
}

fn default_mochat_poll_interval() -> u64 {
    5
}

impl ChannelConfig for MochatConfig {
    fn name() -> &'static str {
        "Mochat"
    }
    fn desc() -> &'static str {
        "Mochat Customer Service"
    }
}

/// Reddit channel configuration (OAuth2 bot).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RedditConfig {
    /// Reddit OAuth2 client ID.
    pub client_id: String,
    /// Reddit OAuth2 client secret.
    pub client_secret: String,
    /// Reddit OAuth2 refresh token for persistent access.
    pub refresh_token: String,
    /// Reddit bot username (without `u/` prefix).
    pub username: String,
    /// Optional subreddit to filter messages (without `r/` prefix).
    /// When set, only messages from this subreddit are processed.
    #[serde(default)]
    pub subreddit: Option<String>,
}

impl ChannelConfig for RedditConfig {
    fn name() -> &'static str {
        "Reddit"
    }
    fn desc() -> &'static str {
        "Reddit bot (OAuth2)"
    }
}

/// Bluesky channel configuration (AT Protocol).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BlueskyConfig {
    /// Bluesky handle (e.g. `"mybot.bsky.social"`).
    pub handle: String,
    /// App-specific password (from Bluesky settings).
    pub app_password: String,
}

impl ChannelConfig for BlueskyConfig {
    fn name() -> &'static str {
        "Bluesky"
    }
    fn desc() -> &'static str {
        "AT Protocol"
    }
}

/// Nostr channel configuration (NIP-04 + NIP-17 private messages)
#[cfg(feature = "channel-nostr")]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NostrConfig {
    /// Private key in hex or nsec bech32 format
    pub private_key: String,
    /// Relay URLs (wss://). Defaults to popular public relays if omitted.
    #[serde(default = "default_nostr_relays")]
    pub relays: Vec<String>,
    /// Allowed sender public keys (hex or npub). Empty = deny all, "*" = allow all
    #[serde(default)]
    pub allowed_pubkeys: Vec<String>,
}

#[cfg(feature = "channel-nostr")]
impl ChannelConfig for NostrConfig {
    fn name() -> &'static str {
        "Nostr"
    }
    fn desc() -> &'static str {
        "Nostr DMs"
    }
}

#[cfg(feature = "channel-nostr")]
pub fn default_nostr_relays() -> Vec<String> {
    vec![
        "wss://relay.damus.io".to_string(),
        "wss://nos.lol".to_string(),
        "wss://relay.primal.net".to_string(),
        "wss://relay.snort.social".to_string(),
    ]
}
