const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatContainer = document.getElementById("chat-container");
const chatHistoryList = document.getElementById("chat-history");
const newChatBtn = document.getElementById("new-chat-btn");

// ====================================
// ACTIVE THREAD
// ====================================
let currentThreadId = null;

// Generate thread id
function generateThreadId() {
    return "thread-" + Date.now();
}

// Clear UI
function clearChat() {
    chatContainer.innerHTML = "";
}

const userIcon = "👤";
const botIcon = "🤖";
const converter = new showdown.Converter();

// ====================================
// Markdown Parser + Table Parser
// ====================================
function formatResponse(text) {
    if (!text) return "";

    const tableRegex = /^\|(.+\|)+$/m;

    if (tableRegex.test(text)) {
        const lines = text.trim().split("\n");
        let html = '<table class="min-w-full border-collapse border border-gray-400 text-sm text-left">';
        let headerParsed = false;

        lines.forEach(line => {
            line = line.trim();
            if (!line.startsWith("|") || !line.endsWith("|")) return;

            if (line.includes("---")) {
                headerParsed = true;
                return;
            }

            const tag = (!headerParsed) ? "th" : "td";
            const cells = line.slice(1, -1).split("|").map(c => c.trim());

            html += "<tr>";
            cells.forEach(c => html += `<${tag} class="border border-gray-300 px-2 py-1">${c}</${tag}>`);
            html += "</tr>";
        });

        html += "</table>";
        return html;
    }

    return converter.makeHtml(text);
}

// ====================================
// Render user message
// ====================================
function renderUserMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-end items-start gap-2";

    const bubble = document.createElement("div");
    bubble.className = "bg-cyan-50 text-gray-900 p-3 rounded-lg max-w-lg";
    bubble.textContent = text;

    const icon = document.createElement("div");
    icon.className = "text-2xl mt-1";
    icon.textContent = userIcon;

    wrapper.appendChild(bubble);
    wrapper.appendChild(icon);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ====================================
// Render tool event
// ====================================
function renderToolEvent(tool, args = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start items-start gap-2";

    const icon = document.createElement("div");
    icon.className = "text-2xl mt-1";
    icon.textContent = "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bg-yellow-100 text-gray-900 p-3 rounded-lg max-w-lg font-mono text-sm";
    
    // Show tool name and args
    let argsDisplay = "";
    if (args && Object.keys(args).length > 0) {
        argsDisplay = `<div class="text-xs mt-1 text-gray-600">Args: ${JSON.stringify(args)}</div>`;
    }
    
    bubble.innerHTML = `<b>Executing tool: ${tool}</b>${argsDisplay}`;

    wrapper.appendChild(icon);
    wrapper.appendChild(bubble);
    chatContainer.appendChild(wrapper);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ====================================
// Render bot message
// ====================================
function renderBotMessage(text) {
    if (!text || text.trim() === "" || text.startsWith("Tool Failed:")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start items-start gap-2";

    const icon = document.createElement("div");
    icon.className = "text-2xl mt-1";
    icon.textContent = botIcon;

    const bubble = document.createElement("div");
    bubble.className = "bg-gray-100 text-gray-900 p-3 rounded-lg max-w-lg prose";
    bubble.innerHTML = formatResponse(text);

    wrapper.appendChild(icon);
    wrapper.appendChild(bubble);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ====================================
// Streaming bot message helpers
// ====================================
let streamingBubble = null;

// Create streaming bubble (correct location)
function createStreamingBubble() {
    // Only create if doesn't exist
    if (streamingBubble) return;

    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start items-start gap-2";

    const icon = document.createElement("div");
    icon.className = "text-2xl mt-1";
    icon.textContent = botIcon;

    const bubble = document.createElement("div");
    bubble.id = "streaming-bot-msg";
    bubble.className = "bg-gray-100 text-gray-900 p-3 rounded-lg max-w-lg prose";
    bubble.innerHTML = "";

    wrapper.appendChild(icon);
    wrapper.appendChild(bubble);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    streamingBubble = bubble;
}

// Smooth render using requestAnimationFrame
let pendingText = "";
let frameBusy = false;

function smoothUpdate(text) {
    pendingText = text;

    if (!frameBusy) {
        frameBusy = true;

        requestAnimationFrame(() => {
            if (streamingBubble) {
                streamingBubble.innerHTML = formatResponse(pendingText);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            frameBusy = false;
        });
    }
}

function finalizeStreamingBubble() {
    streamingBubble = null;
}

// ====================================
// Typing indicator
// ====================================
function showTyping() {
    const div = document.createElement("div");
    div.id = "typing-indicator";
    div.className = "flex justify-start mt-2";

    const bubble = document.createElement("div");
    bubble.className = "bg-gray-100 text-gray-700 p-3 rounded-lg italic";
    bubble.textContent = "Bot is typing...";

    div.appendChild(bubble);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideTyping() {
    const t = document.getElementById("typing-indicator");
    if (t) t.remove();
}

// ====================================
// Load threads
// ====================================
async function loadThreads() {
    try {
        const res = await fetch("http://127.0.0.1:8000/threads");
        const data = await res.json();

        chatHistoryList.innerHTML = "";

        const threads = data.threads.slice().reverse();

        for (const thread of threads) {
            const threadId = thread.thread_id;
            let previewText = thread.summary || "Untitled Chat";

            const row = document.createElement("div");
            row.className =
                `group relative p-3 pl-4 bg-white shadow rounded cursor-pointer 
                hover:bg-gray-200 flex justify-between items-center
                ${threadId === currentThreadId ? "bg-cyan-200 font-semibold" : ""}`;

            const titleEl = document.createElement("div");
            titleEl.textContent = previewText.slice(0, 40) + "...";
            titleEl.className = "truncate flex-1";

            const menuBtn = document.createElement("div");
            menuBtn.className =
                "opacity-0 group-hover:opacity-100 transition cursor-pointer px-2";
            menuBtn.innerHTML = "⋮";

            const dropdown = document.createElement("div");
            dropdown.className =
                "hidden absolute right-2 top-10 bg-white shadow-lg rounded border z-20";
            dropdown.innerHTML =
                `<div class="px-4 py-2 hover:bg-red-100 text-red-600 cursor-pointer">Delete</div>`;

            menuBtn.onclick = (e) => {
                e.stopPropagation();
                dropdown.classList.toggle("hidden");
            };

            dropdown.children[0].onclick = async (e) => {
                e.stopPropagation();
                await deleteThread(threadId);

                if (currentThreadId === threadId) {
                    currentThreadId = null;
                    clearChat();
                }

                loadThreads();
            };

            row.onclick = async () => {
                if (!dropdown.classList.contains("hidden")) return;
                currentThreadId = threadId;
                await loadChatMessages(threadId);
                loadThreads();
            };

            row.appendChild(titleEl);
            row.appendChild(menuBtn);
            row.appendChild(dropdown);

            chatHistoryList.appendChild(row);
        }

    } catch (error) {
        console.error("Failed to load threads:", error);
    }
}

// ====================================
// Load messages
// ====================================
async function loadChatMessages(threadId) {
    clearChat();

    try {
        const res = await fetch(`http://127.0.0.1:8000/thread/${threadId}`);
        const data = await res.json();

        data.messages.forEach(msg => {
            if (msg.type === "HumanMessage") {
                renderUserMessage(msg.content);

            } else if (msg.type === "AIMessage") {
                renderBotMessage(msg.content);

            } else if (msg.type === "ToolEvent") {
                renderToolEvent(msg.tool, msg.args);

            } else if (msg.type === "ToolMessage") {
                // Optionally render tool results
                // renderBotMessage(msg.content);
            }
        });

    } catch (err) {
        console.error("Load messages error:", err);
    }
}

// ====================================
// Delete thread
// ====================================
async function deleteThread(threadId) {
    try {
        await fetch(`http://127.0.0.1:8000/thread/${threadId}`, {
            method: "DELETE"
        });
    } catch (err) {
        console.error("Delete failed:", err);
    }
}

// ====================================
// Send to backend (streaming)
// ====================================
async function sendToBackend(text) {
    return new Promise(async (resolve) => {
        try {
            const response = await fetch("http://127.0.0.1:8000/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    thread_id: currentThreadId,
                    message: text
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let botMessage = "";
            let hasCreatedBubble = false;

            hideTyping();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                let lines = buffer.split("\n");
                buffer = lines.pop(); // keep partial JSON in buffer

                for (let line of lines) {
                    if (!line.startsWith("data:")) continue;

                    try {
                        const payload = JSON.parse(line.slice(5).trim());

                        // Handle tool call
                        if (payload.type === "tool_call") {
                            console.log("Tool call:", payload.tool, payload.args);
                            renderToolEvent(payload.tool, payload.args);
                        }

                        // Handle tool result (optional - you can show or hide this)
                        else if (payload.type === "tool_result") {
                            console.log("Tool result:", payload.tool, payload.result);
                            // Optionally show result:
                            // renderBotMessage(`Tool ${payload.tool} returned: ${payload.result}`);
                        }

                        // Handle text tokens
                        else if (payload.type === "token") {
                            // Create bubble on first token
                            if (!hasCreatedBubble) {
                                createStreamingBubble();
                                hasCreatedBubble = true;
                            }

                            botMessage += payload.token;
                            await new Promise(r => setTimeout(r, 15));
                            smoothUpdate(botMessage);
                        }

                        // Handle done
                        else if (payload.type === "done") {
                            console.log("Stream done");
                            finalizeStreamingBubble();
                        }

                        // Handle error
                        else if (payload.type === "error") {
                            console.error("Stream error:", payload.error);
                            if (!hasCreatedBubble) {
                                createStreamingBubble();
                            }
                            smoothUpdate(`Error: ${payload.error}`);
                            finalizeStreamingBubble();
                        }

                    } catch (err) {
                        console.warn("Bad JSON line:", line, err);
                    }
                }
            }

            // Finalize if not already done
            if (streamingBubble) {
                finalizeStreamingBubble();
            }

            resolve({ answer: botMessage });

        } catch (error) {
            console.error("Streaming error:", error);
            hideTyping();
            renderBotMessage(`Error: ${error.message}`);
            resolve({ answer: "", error: error.message });
        }
    });
}

// ====================================
// Send Message Handler
// ====================================
async function sendMessage() {
    const msg = userInput.value.trim();
    if (!msg) return;

    if (!currentThreadId) {
        currentThreadId = generateThreadId();
        loadThreads();
    }

    renderUserMessage(msg);
    userInput.value = "";
    userInput.focus();

    showTyping();
    await sendToBackend(msg);
    hideTyping();

    loadThreads();
}

// ====================================
// New Chat
// ====================================
newChatBtn.addEventListener("click", () => {
    currentThreadId = generateThreadId();
    clearChat();
    loadThreads();
    renderBotMessage("New chat started. How can I help?");
});

// ====================================
// Events
// ====================================
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ====================================
// Init
// ====================================
async function initializeChat() {
    await loadThreads();

    if (!currentThreadId) {
        currentThreadId = generateThreadId();
        renderBotMessage("Hello! How can I help you?");
    }
}
initializeChat();