# Refactor Plan: `src/channels/mod.rs` (9,108 lines → ~5 modules)

**Status:** Draft  
**Target:** `channels/mod.rs` (largest violation of CLAUDE.md §3.4 SRP)  
**Estimated Effort:** 3-4 weeks (iterative, PR-per-module)

---

## Why This Matters

| CLAUDE.md Says | Current Reality |
|----------------|-----------------|
| "Avoid god modules that mix policy + transport + storage" | 9,108 lines doing everything |
| "Keep each module focused on one concern" | 60+ functions, 8+ responsibilities |
| "One concern per PR" | Everything lives in one file |
| SRP/ISP compliance | `ChannelRuntimeContext` has 25+ fields |

---

## Current File Map

```
src/channels/mod.rs (9,108 lines)
├── Types & Constants (lines 149-399)
├── Provider management (lines 715-792)
├── System prompt building (lines 473-614)
├── Message processing (lines 1228-2148) ← MAIN CULPRIT
├── System prompt loaders (lines 2149-2554)
├── Channel collection (lines 2712+)
└── Inline test implementations (lines 4378+)
```

---

## Proposed Module Structure

```
src/channels/
├── mod.rs                    # Re-exports + thin glue
├── dispatch/                 # NEW: Message routing
│   ├── mod.rs                # run_message_dispatch_loop
│   ├── types.rs              # InFlightSenderTaskState, InFlightTaskCompletion
│   ├── typing.rs             # spawn_scoped_typing_task
│   └── history.rs            # normalize_cached_channel_turns, strip_tool_result
├── processor/                # NEW: Core message processing
│   ├── mod.rs                # process_channel_message (the big one)
│   ├── hooks.rs              # Hook integration
│   ├── memory.rs             # build_memory_context
│   ├── runtime_commands.rs   # handle_runtime_command_if_needed
│   └── tool_context.rs       # extract_tool_context_summary
├── prompt/                    # NEW: System prompt construction
│   ├── mod.rs                # build_channel_system_prompt
│   ├── bootstrap.rs          # load_openclaw_bootstrap_files
│   ├── skills.rs             # replace_available_skills_section
│   └── identity.rs          # Telegram identity helpers
├── provider/                  # NEW: Provider lifecycle
│   ├── mod.rs                # get_or_create_provider
│   └── cache.rs              # Provider cache key, model preview
├── lifecycle/                 # NEW: Channel supervision
│   ├── mod.rs                # spawn_supervised_listener
│   └── backoff.rs            # Backoff calculation
└── ... existing channel impls stay as-is ...
```

---

## Extraction Phases

### Phase 1: Extract Provider Module (Week 1)

**Goal:** Break out `get_or_create_provider` and related logic

**Files to create:**
- `src/channels/provider/mod.rs`
- `src/channels/provider/cache.rs`

**New struct:**
```rust
// src/channels/provider/mod.rs
pub struct ChannelProviderManager {
    ctx: Arc<ChannelRuntimeContext>,
    cache: ProviderCacheMap,
}

impl ChannelProviderManager {
    pub async fn get_or_create(&self, route: &ChannelRouteSelection) -> Result<Arc<dyn Provider>>;
    pub async fn get_or_create_with_key(&self, name: &str, api_key: Option<&str>) -> Result<Arc<dyn Provider>>;
}
```

**Migration steps:**
1. Create `provider/` directory
2. Move constants (`MODEL_CACHE_FILE`, `MODEL_CACHE_PREVIEW_LIMIT`)
3. Move `provider_cache_key`, `load_cached_model_preview`
4. Move `get_or_create_provider`, `create_resilient_provider_nonblocking`
5. Create `ChannelProviderManager` wrapper
6. Update `ChannelRuntimeContext` to hold `ChannelProviderManager`
7. PR: ~300 lines removed from `mod.rs`

---

### Phase 2: Extract Prompt Module (Week 1-2)

**Goal:** Break out system prompt construction

**Files to create:**
- `src/channels/prompt/mod.rs`
- `src/channels/prompt/bootstrap.rs`
- `src/channels/prompt/delivery.rs` (channel_delivery_instructions)

**New struct:**
```rust
// src/channels/prompt/mod.rs
pub struct ChannelPromptBuilder {
    base_prompt: String,
    workspace_dir: PathBuf,
    bootstrap_max_chars: usize,
}

impl ChannelPromptBuilder {
    pub fn build(&self, channel_name: &str, reply_target: &str) -> String;
    pub fn with_skills(&mut self, skills: &str) -> &mut Self;
    pub fn refresh_datetime(&mut self) -> &mut Self;
}
```

**Migration steps:**
1. Create `prompt/` directory
2. Move `channel_delivery_instructions`, `build_channel_system_prompt`
3. Move `load_openclaw_bootstrap_files`, `inject_workspace_file`
4. Move `replace_available_skills_section`, `refreshed_new_session_system_prompt`
5. Move `normalize_telegram_identity`, `bind_telegram_identity`
6. PR: ~500 lines removed

---

### Phase 3: Extract Processor Module (Week 2-3)

**Goal:** Break out `process_channel_message` into composable steps

**Files to create:**
- `src/channels/processor/mod.rs`
- `src/channels/processor/hooks.rs`
- `src/channels/processor/memory.rs`
- `src/channels/processor/commands.rs`

**Key insight:** `process_channel_message` is actually a pipeline. Extract each step:

```rust
// src/channels/processor/mod.rs
pub struct MessageProcessor {
    // Dependencies injected at construction
}

impl MessageProcessor {
    /// Step 1: Validate and trace
    async fn preprocess(&self, msg: traits::ChannelMessage) -> Result<traits::ChannelMessage>;
    
    /// Step 2: Run hooks
    async fn apply_hooks(&self, msg: traits::ChannelMessage) -> Result<Option<traits::ChannelMessage>>;
    
    /// Step 3: Check runtime commands
    async fn handle_runtime_command(&self, msg: &traits::ChannelMessage) -> Result<Option<()>>;
    
    /// Step 4: Build context (provider, memory, history)
    async fn build_context(&self, msg: &traits::ChannelMessage) -> Result<ProcessingContext>;
    
    /// Step 5: Execute the actual processing
    async fn execute(&self, ctx: ProcessingContext) -> Result<()>;
    
    /// Main entry point
    pub async fn process(&self, msg: traits::ChannelMessage) -> Result<()>;
}
```

**Migration steps:**
1. Create `processor/` directory
2. Extract `should_skip_memory_context_entry`, `build_memory_context`
3. Extract hook integration into `hooks.rs`
4. Extract runtime command handling into `commands.rs`
5. Extract `process_channel_message` piece by piece
6. PR: ~800 lines removed

---

### Phase 4: Extract Dispatch Module (Week 3-4)

**Goal:** Break out message dispatch orchestration

**Files to create:**
- `src/channels/dispatch/mod.rs` (already exists, expand it)
- `src/channels/dispatch/types.rs`
- `src/channels/dispatch/typing.rs`
- `src/channels/dispatch/history.rs`

**Migration steps:**
1. Move `InFlightSenderTaskState`, `InFlightTaskCompletion` to `types.rs`
2. Move `spawn_scoped_typing_task` to `typing.rs`
3. Move history normalization to `history.rs`
4. Move `spawn_supervised_listener`, `spawn_supervised_listener_with_health_interval`
5. PR: ~300 lines removed

---

## Backward Compatibility Strategy

### Keep `mod.rs` as Thin Re-export

```rust
// src/channels/mod.rs — AFTER refactor

// Re-export from new modules for internal use
pub(crate) mod dispatch;
pub(crate) mod processor;
pub(crate) mod prompt;
pub(crate) mod provider;

// Keep only:
// 1. pub fn build_system_prompt (public API)
// 2. ChannelRuntimeContext struct (needs refactoring)
// 3. Constants that are truly shared
// 4. The Channel trait (stays here)

// Target: ~1000 lines
```

### Deprecation Path

```rust
// Old code still works:
use crate::channels::process_channel_message;

// After refactor:
use crate::channels::processor::process_message;

// Create alias during transition:
#[deprecated(since = "0.6.0", note = "Use channels::processor::process_message")]
pub use processor::process_message as process_channel_message;
```

---

## Validation Checklist

For each PR, verify:

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test --lib`
- [ ] No new imports into `channels/mod.rs`
- [ ] New module has its own tests
- [ ] Benchmarks still pass

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing channel implementations | Phase 1 only touches provider, no channel code changes |
| Refactor in flight blocks new features | Work on isolated modules, merge incrementally |
| Performance regression | Profile before/after with `rain-bench` |
| Missing edge cases in extracted code | Copy tests first, then refactor |

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| `channels/mod.rs` lines | 9,108 | ~1,000 |
| Single function max length | ~800 | ~150 |
| Modules with single responsibility | 1 | 6+ |
| Time to understand message flow | Hours | Minutes |

---

## Alternative: Don't Refactor (Honest Option)

**Arguments for leaving it:**
- It works. 4,630 tests pass.
- Refactoring introduces risk
- The trait boundaries are actually good
- The "god module" is an organizational issue, not a correctness one

**Arguments for refactoring:**
- Onboarding new contributors is harder
- Testing specific concerns is harder
- CLAUDE.md is aspirational but not enforced
- Technical debt compounds

**Recommendation:** Do Phase 1 (provider extraction) only — it's the cleanest boundary and lowest risk. The rest can wait until there's actual pain.

---

## Next Steps

1. Create branch: `refactor/channels-extract-provider`
2. Create `src/channels/provider/` directory
3. Move code incrementally (one function family at a time)
4. Update imports, run tests
5. Open PR, review, merge
6. Repeat for next phase

**Want me to start Phase 1?** I'd create the `provider/` module and do the first extraction.
