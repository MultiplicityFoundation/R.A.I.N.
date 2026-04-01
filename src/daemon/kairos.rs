use tokio::time::{sleep, Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, error};

use crate::memory::sqlite::SqliteMemory;
use crate::memory::knowledge_graph::KnowledgeGraph;

#[derive(Serialize)]
struct KairosPayload {
    memories: Vec<crate::memory::sqlite::MemoryRow>,
}

#[derive(Deserialize)]
struct KnowledgeNode {
    entity: String,
    relationship: String,
    target: String,
    context: String,
}

#[derive(Deserialize)]
struct KairosResponse {
    compressed_nodes: Vec<KnowledgeNode>,
}

pub struct KairosDaemon {
    memory_db: SqliteMemory,
    graph_db: KnowledgeGraph,
    idle_threshold: Duration,
}

impl KairosDaemon {
    pub fn new(memory_db: SqliteMemory, graph_db: KnowledgeGraph) -> Self {
        Self {
            memory_db,
            graph_db,
            idle_threshold: Duration::from_secs(300), // 5 min idle
        }
    }

    pub async fn run_background_loop(self) {
        info!("KAIROS Dreaming Daemon initialized.");
        
        loop {
            sleep(Duration::from_secs(60)).await;

            // 1. Check if we have unconsolidated memories
            let memories = match self.memory_db.fetch_unconsolidated(50).await {
                Ok(m) if !m.is_empty() => m,
                _ => continue,
            };

            // 2. Check system idle state (using your watermark from Task 1)
            if let Ok(Some(last_write)) = self.memory_db.latest_episodic_write_at().await {
                let elapsed = chrono::Utc::now().timestamp() - last_write;
                if elapsed < self.idle_threshold.as_secs() as i64 {
                    continue; 
                }
            }

            debug!("System idle. KAIROS entering REM state to process {} memories...", memories.len());

            // 3. Send to Python Dreamer
            let payload = KairosPayload { memories: memories.clone() };
            if let Ok(response) = self.call_python_dreamer(&payload).await {
                // 4. Inject nodes into Knowledge Graph (fallback)
                for node in response.compressed_nodes {
                    let relation = crate::memory::knowledge_graph::Relation::from_dreamer(&node.relationship);
                    let _ = self.graph_db.get_or_create_node(
                        &node.entity, "Concept", &node.context, &node.target, "kairos"
                    );
                }

                // 5. Mark as consolidated
                let ids: Vec<i64> = memories.into_iter().map(|m| m.id).collect();
                let _ = self.memory_db.mark_as_consolidated(&ids).await;
                info!("KAIROS dreaming sequence complete.");
            }
        }
    }

    #[cfg(unix)]
    async fn call_python_dreamer(&self, payload: &KairosPayload) -> anyhow::Result<KairosResponse> {
        use tokio::net::UnixStream;
        let mut stream = UnixStream::connect("/tmp/kairos_dreamer.sock").await?;
        self.transact(&mut stream, payload).await
    }

    #[cfg(windows)]
    async fn call_python_dreamer(&self, payload: &KairosPayload) -> anyhow::Result<KairosResponse> {
        use tokio::net::TcpStream;
        let mut stream = TcpStream::connect("127.0.0.1:50051").await?;
        self.transact(&mut stream, payload).await
    }

    async fn transact<S: AsyncReadExt + AsyncWriteExt + Unpin>(&self, stream: &mut S, payload: &KairosPayload) -> anyhow::Result<KairosResponse> {
        let json_payload = serde_json::to_string(payload)?;
        stream.write_all(json_payload.as_bytes()).await?;
        
        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).await?;
        
        let response: KairosResponse = serde_json::from_slice(&buffer)?;
        Ok(response)
    }
}
