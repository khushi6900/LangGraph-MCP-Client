const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatContainer = document.getElementById("chat-container");
const chatHistoryList = document.getElementById("chat-history");
const newChatBtn = document.getElementById("new-chat-btn");
const uploadBtn = document.getElementById("file-upload-btn");
const hiddenInput = document.getElementById("hidden-file-input");
const uploadedFilesContainer = document.getElementById("uploaded-files");
const uploadStatus = document.getElementById("upload-status");

// ====================================
// ACTIVE THREAD
// ====================================

const API_BASE = "http://127.0.0.1:8001";

let currentThreadId = null;

function generateThreadId() {
    return "thread-" + Date.now();
}

function clearChat() {
    chatContainer.innerHTML = "";
}

const converter = new showdown.Converter();

function makeAvatar(isUser) {
    const a = document.createElement("div");
    a.className = "w-7 h-7 shrink-0 rounded-full flex items-center justify-center text-[11px] font-semibold " +
        (isUser ? "bg-slate-200 text-slate-600" : "bg-indigo-100 text-indigo-700");
    a.textContent = isUser ? "You" : "AI";
    return a;
}

uploadBtn.addEventListener("click", () => {
    hiddenInput.click();
});

hiddenInput.addEventListener("change", async () => {
    const file = hiddenInput.files[0];
    if (!file) return;

    uploadStatus.innerHTML = `<span class="text-indigo-600">Uploading "${file.name}"...</span>`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const uploadUrl = `${API_BASE}/upload/${currentThreadId}`;

        const res = await fetch(uploadUrl, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        console.log("UPLOAD RESPONSE:", data);

        addUploadedFileToUI(file.name, data.file_id, data.active_file_id);

        uploadStatus.innerHTML = `
            <span class="text-emerald-600">
                Uploaded: ${file.name} (Vectors: ${data.total_vectors})
            </span>
        `;

        setTimeout(() => (uploadStatus.innerHTML = ""), 3000);

    } catch (err) {
        uploadStatus.innerHTML = `<span class="text-rose-600">Upload failed!</span>`;
        console.error("Upload failed", err);
    }
});

function addUploadedFileToUI(filename, fileId, activeFileId) {

    const fileItem = document.createElement("div");
    fileItem.className =
        "px-2.5 py-1.5 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-700 flex justify-between items-center gap-2";

    const isActive = fileId === activeFileId;

    fileItem.innerHTML = `
        <span class="truncate">${filename}</span>
        ${
            isActive
                ? `<span class="text-emerald-600 font-medium text-[11px]">(Active)</span>`
                : `<span class="text-slate-400 text-[11px]">Uploaded</span>`
        }
    `;

    uploadedFilesContainer.appendChild(fileItem);
}

// ====================================
// Markdown Parser + Table Parser
// ====================================
function formatResponse(text) {
    if (!text) return "";

    const tableRegex = /^\|(.+\|)+$/m;

    if (tableRegex.test(text)) {
        const lines = text.trim().split("\n");
        let html = '<table class="min-w-full border-collapse border border-slate-200 text-sm text-left">';
        let headerParsed = false;

        lines.forEach(line => {
            line = line.trim();
            if (!line.startsWith("|") || !line.endsWith("|")) return;

            if (line.includes("---")) {
                headerParsed = true;
                return;
            }

            const tag = (!headerParsed) ? "th" : "td";
            const extra = tag === "th" ? " bg-slate-50 font-medium" : "";
            const cells = line.slice(1, -1).split("|").map(c => c.trim());

            html += "<tr>";
            cells.forEach(c => html += `<${tag} class="border border-slate-200 px-2 py-1${extra}">${c}</${tag}>`);
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
    wrapper.className = "flex justify-end items-start gap-3";

    const bubble = document.createElement("div");
    bubble.className = "bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-xl text-sm leading-relaxed";
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    wrapper.appendChild(makeAvatar(true));

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ====================================
// Render tool event
// ====================================
function renderToolEvent(tool, args = {}) {

    // Only show executing message for these tools
    const allowedTools = ["add_expense", "list_expenses"];

    if (!allowedTools.includes(tool)) {
        console.log(`Skipping tool execution bubble for: ${tool}`);
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start items-start gap-3";

    const bubble = document.createElement("div");
    bubble.className =
        "bg-slate-100 border border-slate-200 text-slate-600 px-3 py-2 rounded-lg max-w-xl font-mono text-xs";

    let argsDisplay = "";
    if (args && Object.keys(args).length > 0) {
        argsDisplay = `<div class="text-xs mt-1 text-slate-500">
            Args: ${JSON.stringify(args)}
        </div>`;
    }

    bubble.innerHTML = `<b>Executing tool: ${tool}</b>${argsDisplay}`;

    wrapper.appendChild(makeAvatar(false));
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
    wrapper.className = "flex justify-start items-start gap-3";

    const bubble = document.createElement("div");
    bubble.className = "bg-white border border-slate-200 text-slate-800 px-4 py-2.5 rounded-2xl rounded-tl-sm max-w-xl text-sm leading-relaxed prose prose-sm prose-slate";
    bubble.innerHTML = formatResponse(text);

    wrapper.appendChild(makeAvatar(false));
    wrapper.appendChild(bubble);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ====================================
// Streaming bot message helpers
// ====================================
let streamingBubble = null;

function createStreamingBubble() {
    if (streamingBubble) return;

    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start items-start gap-3";

    const bubble = document.createElement("div");
    bubble.id = "streaming-bot-msg";
    bubble.className = "bg-white border border-slate-200 text-slate-800 px-4 py-2.5 rounded-2xl rounded-tl-sm max-w-xl text-sm leading-relaxed prose prose-sm prose-slate";
    bubble.innerHTML = "";

    wrapper.appendChild(makeAvatar(false));
    wrapper.appendChild(bubble);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    streamingBubble = bubble;
}

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
    bubble.className = "bg-white border border-slate-200 text-slate-500 px-4 py-2 rounded-2xl text-sm";
    bubble.textContent = "Thinking…";

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
        const res = await fetch(`${API_BASE}/threads`);
        const data = await res.json();

        chatHistoryList.innerHTML = "";

        const threads = data.threads.slice().reverse();

        for (const thread of threads) {
            const threadId = thread.thread_id;
            let previewText = thread.summary || "Untitled Chat";

            const row = document.createElement("div");
            row.className = `group relative px-3 py-2 rounded-lg cursor-pointer text-sm flex justify-between items-center transition ${
                threadId === currentThreadId
                    ? "bg-slate-100 text-slate-900 font-medium"
                    : "text-slate-600 hover:bg-slate-50"}`;

            const titleEl = document.createElement("div");
            titleEl.textContent = previewText.slice(0, 40) + "...";
            titleEl.className = "truncate flex-1";

            const menuBtn = document.createElement("div");
            menuBtn.className =
                "opacity-0 group-hover:opacity-100 transition cursor-pointer px-2 text-slate-400 hover:text-slate-700";
            menuBtn.innerHTML = "⋮";

            const dropdown = document.createElement("div");
            dropdown.className =
                "hidden absolute right-2 top-9 bg-white shadow-md rounded-lg border border-slate-200 text-sm z-20";
            dropdown.innerHTML =
                `<div class="px-4 py-2 hover:bg-rose-50 text-rose-600 cursor-pointer rounded-lg">Delete</div>`;

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
        const res = await fetch(`${API_BASE}/thread/${threadId}`);
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
        await fetch(`${API_BASE}/thread/${threadId}`, {
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
            const response = await fetch(`${API_BASE}/chat/stream`, {
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

                        if (payload.type === "tool_call") {
                            console.log("Tool call:", payload.tool, payload.args);
                            renderToolEvent(payload.tool, payload.args);
                        }

                        else if (payload.type === "tool_result") {
                            console.log("Tool result:", payload.tool, payload.result);
                        }

                        else if (payload.type === "token") {
                            hideTyping();
                            if (!hasCreatedBubble) {
                                createStreamingBubble();
                                hasCreatedBubble = true;
                            }

                            botMessage += payload.token;
                            await new Promise(r => setTimeout(r, 15));
                            smoothUpdate(botMessage);
                        }

                        else if (payload.type === "done") {
                            console.log("Stream done");
                            finalizeStreamingBubble();
                        }

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

            if (buffer.trim() !== "") {
                try {
                    const payload = JSON.parse(buffer.replace("data:", "").trim());
                    if (payload.type === "token") {
                        botMessage += payload.token;
                        smoothUpdate(botMessage);
                    }
                } catch (e) {
                    console.warn("Failed to parse final buffer:", buffer, e);
                }
            }

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