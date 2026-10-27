// Récupération des éléments du DOM pour les onglets, conteneur de chat et bouton
const chatTabs = document.getElementById('chat-tabs');
const chatContainer = document.getElementById('chat-container');
const newTabButton = document.getElementById('new-tab-button');

// Initialisation des variables
let chatHistory = {};               // Historique de chat stocké par session
let lastQuestion = '';              // Dernière question posée
let lastResponse = '';              // Dernière réponse obtenue
let currentSessionId = 'default';   // ID de session en cours

// URL de l'API Flask
const API_URL = 'http://localhost:5000';

// Génère un ID de session unique
function generateSessionId() {
    return Math.random().toString(36).substring(2, 15) +
           Math.random().toString(36).substring(2, 15);
}

// Détecte la langue du texte (français par défaut, arabe ou anglais)
function detectLang(text) {
    const arabic = /[\u0600-\u06FF]/;
    const english = /^[a-zA-Z0-9\s.,?!'"()]+$/;

    if (arabic.test(text)) return 'ar';
    if (english.test(text)) return 'en';
    return 'fr'; // Par défaut
}

// Crée une fenêtre de chat pour une session donnée
function createChatWindow(sessionId) {
    const chatWindow = document.createElement('div');
    chatWindow.className = 'chat-window';
    chatWindow.dataset.sessionId = sessionId;
    chatWindow.innerHTML = `
        <h2>Chatbot ISET</h2>
        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Posez votre question...">
            <button class="send-button">Envoyer</button>
        </div>
        <div id="feedback-section" class="feedback-section"></div>
    `;
    chatContainer.appendChild(chatWindow);

    const sendButton = chatWindow.querySelector('.send-button');
    const userInput = chatWindow.querySelector('#user-input');

    if (sendButton) {
        sendButton.addEventListener('click', () => {
            sendMessage(sessionId, userInput);
        });
    }

    if (userInput) {
        userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                sendMessage(sessionId, userInput);
            }
        });
    }

    return chatWindow;
}

// Crée un onglet pour chaque session de conversation
function createChatTab(sessionId, tabName = 'Conversation') {
    const tab = document.createElement('button');
    tab.className = 'tab';
    tab.dataset.sessionId = sessionId;
    tab.textContent = tabName;
    chatTabs.insertBefore(tab, newTabButton);
    tab.addEventListener('click', () => switchChat(sessionId));
    return tab;
}

// Permute entre les différentes sessions (onglets) de chat
function switchChat(sessionId) {
    document.querySelectorAll('.chat-window').forEach(window => window.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

    let chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    let tab = document.querySelector(`.tab[data-session-id="${sessionId}"]`);

    if (!chatWindow) {
        chatWindow = createChatWindow(sessionId);
    }
    chatWindow.classList.add('active');

    if (!tab) {
        tab = createChatTab(sessionId, 'Nouvelle conversation');
    }
    tab.classList.add('active');

    currentSessionId = sessionId;
    loadChatHistory(sessionId);
}

// Fonction pour récupérer une réponse du modèle ML (si vous avez un endpoint dédié)
async function getMLResponse(message) {
    try {
        const response = await fetch(`${API_URL}/ml_response`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: message })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
            return null;
        }
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error("Erreur lors de la récupération de la réponse ML :", error);
        return "Erreur lors de la communication avec le modèle ML.";
    }
}

// Crée une nouvelle session au clic sur "+ Nouvelle conversation"
newTabButton.addEventListener('click', () => {
    const newSessionId = generateSessionId();
    createChatWindow(newSessionId);
    createChatTab(newSessionId, 'Nouvelle conversation');
    switchChat(newSessionId);
});

// Ajoute un message (utilisateur ou bot) dans le chat
function addMessage(message, sender, sessionId) {
    const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    if (!chatWindow) return;

    const chatBox = chatWindow.querySelector('#chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    msgDiv.textContent = message;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Sauvegarde dans l’historique
    if (!chatHistory[sessionId]) {
        chatHistory[sessionId] = [];
    }
    chatHistory[sessionId].push({ text: message, sender: sender });
    saveChatHistory(sessionId);
}

// Envoie un message à l'API Flask
function sendMessage(sessionId, inputElement) {
    const question = inputElement.value.trim();
    if (!question) return;

    const lang = detectLang(question); // Détection automatique de la langue
    addMessage(question, 'user', sessionId);
    inputElement.value = '';

    const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    const feedbackSection = chatWindow.querySelector('#feedback-section');
    feedbackSection.innerHTML = '';

    fetch(`${API_URL}/chatbot`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Session-Id': sessionId
        },
        body: JSON.stringify({ question: question, lang: lang, session_id: sessionId })
    })
    .then(res => {
        if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
        return res.json();
    })
    .then(data => {
        addMessage(data.response, 'bot', sessionId);
        lastQuestion = data.original_question;
        lastResponse = data.response;
        localStorage.setItem('sessionId', sessionId);
        showFeedbackOptions(sessionId);
    })
    .catch(error => {
        console.error("Erreur lors de l'envoi :", error);
        addMessage("Erreur serveur. Veuillez réessayer.", 'bot', sessionId);
    });
}

// Affiche les options de feedback (👍 👎)
function showFeedbackOptions(sessionId) {
    const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    const feedbackSection = chatWindow.querySelector('#feedback-section');
    feedbackSection.innerHTML = `
        <span>Cette réponse vous a-t-elle été utile ? </span>
        <button onclick="sendFeedback('${sessionId}', 'like')">👍 Oui</button>
        <button onclick="sendFeedback('${sessionId}', 'dislike')">👎 Non</button>
    `;
}

// Envoie un feedback (like/dislike) au backend
function sendFeedback(sessionId, feedback) {
    fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Session-Id': sessionId
        },
        body: JSON.stringify({
            question: lastQuestion,
            response: lastResponse,
            feedback: feedback
        })
    })
    .then(res => res.json())
    .then(data => {
        const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
        const feedbackSection = chatWindow.querySelector('#feedback-section');
        feedbackSection.innerHTML = `<p>Merci pour votre retour !</p>`;
    })
    .catch(error => {
        console.error("Erreur feedback :", error);
        const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
        const feedbackSection = chatWindow.querySelector('#feedback-section');
        feedbackSection.innerHTML = `<p>Erreur lors de l’envoi du feedback.</p>`;
    });
}

// Charge la session par défaut à l'ouverture de la page
window.onload = function () {
    switchChat(currentSessionId); // Charge la session par défaut
};

// Charge l'historique depuis localStorage et le serveur
function loadChatHistory(sessionId) {
    const savedHistory = localStorage.getItem(`chatHistory_${sessionId}`);
    if (savedHistory) {
        try {
            chatHistory[sessionId] = JSON.parse(savedHistory);
        } catch (error) {
            console.error("Erreur lors de la parse de l'historique local :", error);
            localStorage.removeItem(`chatHistory_${sessionId}`); // Nettoyer si corrompu
            chatHistory[sessionId] = [];
        }
    } else {
        chatHistory[sessionId] = [];
    }

    const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    if (!chatWindow) return;
    const chatBox = chatWindow.querySelector('#chat-box');
    chatBox.innerHTML = '';

    if (chatHistory[sessionId]) {
        chatHistory[sessionId].forEach(msg => {
            addMessage(msg.text, msg.sender, sessionId);
        });
    }

    // Historique depuis le serveur (optionnel)
    fetch(`${API_URL}/history/${sessionId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-Session-Id': sessionId
        },
    })
    .then(res => res.json())
    .then(history => {
        if (!chatHistory[sessionId]) chatHistory[sessionId] = [];
        history.forEach(msg => {
            const exists = chatHistory[sessionId].find(m => m.text === msg.text && m.sender === msg.sender);
            if (!exists) {
                chatHistory[sessionId].push(msg);
                addMessage(msg.text, msg.sender, sessionId);
            }
        });
        saveChatHistory(sessionId);
    })
    .catch(error => console.error('Erreur chargement historique:', error));
}

// Sauvegarde l’historique dans le localStorage
function saveChatHistory(sessionId) {
    localStorage.setItem(`chatHistory_${sessionId}`, JSON.stringify(chatHistory[sessionId]));
}

// Exemple d'utilisation de getMLResponse (à intégrer dans votre logique de sendMessage si nécessaire)
async function handleUserInput(userMessage, sessionId) {
    addMessage(userMessage, 'user', sessionId);
    const mlResponse = await getMLResponse(userMessage);
    if (mlResponse) {
        console.log("Réponse prédite ML :", mlResponse);
        addMessage(mlResponse, 'bot-ml', sessionId); // Afficher la réponse ML dans le chat
    } else {
        // Gérer le cas où la réponse ML est nulle ou en cas d'erreur
        addMessage("Erreur lors de la récupération de la réponse ML.", 'bot-error', sessionId);
    }
}

// Modifier la fonction sendMessage pour potentiellement utiliser getMLResponse
// Exemple d'intégration (vous devrez adapter cela à votre logique)
function sendMessage(sessionId, inputElement) {
    const question = inputElement.value.trim();
    if (!question) return;

    const lang = detectLang(question);
    addMessage(question, 'user', sessionId);
    inputElement.value = '';

    const chatWindow = document.querySelector(`.chat-window[data-session-id="${sessionId}"]`);
    const feedbackSection = chatWindow.querySelector('#feedback-section');
    feedbackSection.innerHTML = '';

    // Choisir d'utiliser l'API Flask principale ou l'endpoint ML dédié
    const useMLResponse = false; // Définir à true si vous voulez utiliser uniquement l'endpoint ML

    if (useMLResponse) {
        handleUserInput(question, sessionId);
    } else {
        fetch(`${API_URL}/chatbot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-Id': sessionId
            },
            body: JSON.stringify({ question: question, lang: lang, session_id: sessionId })
        })
        .then(res => {
            if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
            return res.json();
        })
        .then(data => {
            addMessage(data.response, 'bot', sessionId);
            lastQuestion = data.original_question;
            lastResponse = data.response;
            localStorage.setItem('sessionId', sessionId);
            showFeedbackOptions(sessionId);
        })
        .catch(error => {
            console.error("Erreur lors de l'envoi :", error);
            addMessage("Erreur serveur. Veuillez réessayer.", 'bot', sessionId);
        });
    }
}