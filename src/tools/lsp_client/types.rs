use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use lsp_types::{Diagnostic, Range, SymbolKind};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub workspace_root: PathBuf,
    pub initialization_options: Option<Value>,
    pub extension_to_language: BTreeMap<String, String>,
}

impl LspServerConfig {
    #[must_use]
    pub fn language_id_for(&self, path: &Path) -> Option<&str> {
        let extension = normalize_extension(path.extension()?.to_string_lossy().as_ref());
        self.extension_to_language
            .get(&extension)
            .map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileDiagnostics {
    pub path: PathBuf,
    pub uri: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct WorkspaceDiagnostics {
    pub files: Vec<FileDiagnostics>,
}

impl WorkspaceDiagnostics {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    #[must_use]
    pub fn total_diagnostics(&self) -> usize {
        self.files.iter().map(|file| file.diagnostics.len()).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SymbolLocation {
    pub path: PathBuf,
    pub range: Range,
}

impl SymbolLocation {
    #[must_use]
    pub fn start_line(&self) -> u32 {
        self.range.start.line + 1
    }

    #[must_use]
    pub fn start_character(&self) -> u32 {
        self.range.start.character + 1
    }
}

impl Display for SymbolLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.path.display(),
            self.start_line(),
            self.start_character()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DocumentSymbolEntry {
    pub name: String,
    pub detail: Option<String>,
    pub kind: SymbolKind,
    pub path: PathBuf,
    pub range: Range,
    pub selection_range: Range,
    pub container_name: Option<String>,
    pub children: Vec<DocumentSymbolEntry>,
}

impl DocumentSymbolEntry {
    fn from_document_symbol(
        symbol: lsp_types::DocumentSymbol,
        path: PathBuf,
        container_name: Option<String>,
    ) -> Self {
        let children = symbol
            .children
            .unwrap_or_default()
            .into_iter()
            .map(|child| Self::from_document_symbol(child, path.clone(), Some(symbol.name.clone())))
            .collect();

        Self {
            name: symbol.name,
            detail: symbol.detail,
            kind: symbol.kind,
            path,
            range: symbol.range,
            selection_range: symbol.selection_range,
            container_name,
            children,
        }
    }

    fn from_symbol_information(symbol: lsp_types::SymbolInformation) -> Option<Self> {
        Some(Self {
            name: symbol.name,
            detail: None,
            kind: symbol.kind,
            path: url::Url::parse(symbol.location.uri.as_str())
                .ok()?
                .to_file_path()
                .ok()?,
            range: symbol.location.range,
            selection_range: symbol.location.range,
            container_name: symbol.container_name,
            children: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LspContextEnrichment {
    pub file_path: PathBuf,
    pub diagnostics: WorkspaceDiagnostics,
    pub definitions: Vec<SymbolLocation>,
    pub references: Vec<SymbolLocation>,
}

#[must_use]
pub(crate) fn normalize_extension(extension: &str) -> String {
    if extension.starts_with('.') {
        extension.to_ascii_lowercase()
    } else {
        format!(".{}", extension.to_ascii_lowercase())
    }
}

pub(crate) fn symbol_response_to_entries(
    path: &Path,
    response: Option<lsp_types::DocumentSymbolResponse>,
) -> Vec<DocumentSymbolEntry> {
    match response {
        Some(lsp_types::DocumentSymbolResponse::Nested(symbols)) => symbols
            .into_iter()
            .map(|symbol| {
                DocumentSymbolEntry::from_document_symbol(symbol, path.to_path_buf(), None)
            })
            .collect(),
        Some(lsp_types::DocumentSymbolResponse::Flat(symbols)) => symbols
            .into_iter()
            .filter_map(DocumentSymbolEntry::from_symbol_information)
            .collect(),
        None => Vec::new(),
    }
}
