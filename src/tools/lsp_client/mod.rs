mod client;
mod error;
mod manager;
mod types;

pub use manager::LspManager;
pub use types::{DocumentSymbolEntry, LspServerConfig, SymbolLocation};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    use lsp_types::Position;

    use super::{LspManager, LspServerConfig};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("rain-lsp-{label}-{nanos}"))
    }

    fn python_command() -> Option<String> {
        ["python", "python3"].into_iter().find_map(|candidate| {
            Command::new(candidate)
                .arg("--version")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|_| candidate.to_string())
        })
    }

    fn write_mock_server_script(root: &std::path::Path) -> PathBuf {
        let script_path = root.join("mock_lsp_server.py");
        fs::write(
            &script_path,
            r#"import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b"\r\n":
            break
        key, value = line.decode("utf-8").split(":", 1)
        headers[key.lower()] = value.strip()
    length = int(headers["content-length"])
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def write_message(payload):
    raw = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break

    method = message.get("method")
    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "capabilities": {
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "documentSymbolProvider": True,
                    "textDocumentSync": 1,
                }
            },
        })
    elif method == "initialized":
        continue
    elif method == "textDocument/didOpen":
        continue
    elif method == "textDocument/documentSymbol":
        uri = message["params"]["textDocument"]["uri"]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": [
                {
                    "name": "main",
                    "kind": 12,
                    "location": {
                        "uri": uri,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 11},
                        }
                    }
                }
            ],
        })
    elif method == "textDocument/definition":
        uri = message["params"]["textDocument"]["uri"]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": [
                {
                    "uri": uri,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                }
            ],
        })
    elif method == "textDocument/references":
        uri = message["params"]["textDocument"]["uri"]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": [
                {
                    "uri": uri,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                },
                {
                    "uri": uri,
                    "range": {
                        "start": {"line": 1, "character": 4},
                        "end": {"line": 1, "character": 8},
                    },
                },
            ],
        })
    elif method == "shutdown":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": None})
    elif method == "exit":
        break
"#,
        )
        .expect("mock server should be written");
        script_path
    }

    #[tokio::test(flavor = "current_thread")]
    async fn manager_supports_symbols_definitions_and_references() {
        let Some(python) = python_command() else {
            return;
        };

        let root = temp_dir("manager");
        fs::create_dir_all(root.join("src")).expect("workspace root should exist");
        let script_path = write_mock_server_script(&root);
        let source_path = root.join("src").join("main.rs");
        fs::write(&source_path, "fn main() {}\nmain();\n").expect("source file should exist");

        let manager = LspManager::new(vec![LspServerConfig {
            name: "rust-analyzer".to_string(),
            command: python,
            args: vec![script_path.display().to_string()],
            env: BTreeMap::new(),
            workspace_root: root.clone(),
            initialization_options: None,
            extension_to_language: BTreeMap::from([(".rs".to_string(), "rust".to_string())]),
        }])
        .expect("manager should build");

        let symbols = manager
            .document_symbols(&source_path)
            .await
            .expect("document symbols request should succeed");
        let definitions = manager
            .go_to_definition(&source_path, Position::new(0, 0))
            .await
            .expect("definition request should succeed");
        let references = manager
            .find_references(&source_path, Position::new(0, 0), true)
            .await
            .expect("references request should succeed");

        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "main");
        assert_eq!(definitions.len(), 1);
        assert_eq!(references.len(), 2);

        manager.shutdown().await.expect("shutdown should succeed");
        fs::remove_dir_all(root).expect("temp workspace should be removed");
    }
}
