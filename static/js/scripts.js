// creating a chat application there will be functions needed to append the chat messages to the chat box, send messages, and retrieve messages from the server.

var chatbox = document.getElementById("chat-box")

// HTML of the chat box is as follows 

// = """<!-- <div class="chat__conversation-board">
//           <div class="chat__conversation-board__message-container">
//             <div class="chat__conversation-board__message__person">
//               <div class="chat__conversation-board__message__person__avatar"><img src="https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0" alt="Monika Figi"/></div><span class="chat__conversation-board__message__person__nickname">Monika Figi</span>
//             </div>
//             <div class="chat__conversation-board__message__context">
//               <div class="chat__conversation-board__message__bubble"> <span>Somewhere stored deep, deep in my memory banks is the phrase &quot;It really whips the llama's ass&quot;.</span></div>
//             </div>
//             <div class="chat__conversation-board__message__options">
//               <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
//                 <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="10"></circle>
//                   <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
//                   <line x1="9" y1="9" x2="9.01" y2="9"></line>
//                   <line x1="15" y1="9" x2="15.01" y2="9"></line>
//                 </svg>
//               </button>
//               <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
//                 <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="1"></circle>
//                   <circle cx="19" cy="12" r="1"></circle>
//                   <circle cx="5" cy="12" r="1"></circle>
//                 </svg>
//               </button>
//             </div>
//           </div>
//           <div class="chat__conversation-board__message-container">
//             <div class="chat__conversation-board__message__person">
//               <div class="chat__conversation-board__message__person__avatar"><img src="https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0" alt="Thomas Rogh"/></div><span class="chat__conversation-board__message__person__nickname">Thomas Rogh</span>
//             </div>
//             <div class="chat__conversation-board__message__context">
//               <div class="chat__conversation-board__message__bubble"> <span>Think the guy that did the voice has a Twitter?</span></div>
//             </div>
//             <div class="chat__conversation-board__message__options">
//               <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
//                 <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="10"></circle>
//                   <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
//                   <line x1="9" y1="9" x2="9.01" y2="9"></line>
//                   <line x1="15" y1="9" x2="15.01" y2="9"></line>
//                 </svg>
//               </button>
//               <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
//                 <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="1"></circle>
//                   <circle cx="19" cy="12" r="1"></circle>
//                   <circle cx="5" cy="12" r="1"></circle>
//                 </svg>
//               </button>
//             </div>
//           </div>
//           <div class="chat__conversation-board__message-container">
//             <div class="chat__conversation-board__message__person">
//               <div class="chat__conversation-board__message__person__avatar"><img src="https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0" alt="Monika Figi"/></div><span class="chat__conversation-board__message__person__nickname">Monika Figi</span>
//             </div>
//             <div class="chat__conversation-board__message__context">
//               <div class="chat__conversation-board__message__bubble"> <span>WE MUST FIND HIM!!</span></div>
//               <div class="chat__conversation-board__message__bubble"> <span>Wait ...</span></div>
//             </div>
//             <div class="chat__conversation-board__message__options">
//               <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
//                 <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="10"></circle>
//                   <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
//                   <line x1="9" y1="9" x2="9.01" y2="9"></line>
//                   <line x1="15" y1="9" x2="15.01" y2="9"></line>
//                 </svg>
//               </button>
//               <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
//                 <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="1"></circle>
//                   <circle cx="19" cy="12" r="1"></circle>
//                   <circle cx="5" cy="12" r="1"></circle>
//                 </svg>
//               </button>
//             </div>
//           </div>
//           <div class="chat__conversation-board__message-container reversed">
//             <div class="chat__conversation-board__message__person">
//               <div class="chat__conversation-board__message__person__avatar"><img src="https://th.bing.com/th/id/R.8e2c571ff125b3531705198a15d3103c?rik=gzhbzBpXBa%2bxMA&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fuser-png-icon-big-image-png-2240.png&ehk=VeWsrun%2fvDy5QDv2Z6Xm8XnIMXyeaz2fhR3AgxlvxAc%3d&risl=&pid=ImgRaw&r=0" alt="Dennis Mikle"/></div><span class="chat__conversation-board__message__person__nickname">Dennis Mikle</span>
//             </div>
//             <div class="chat__conversation-board__message__context">
//               <div class="chat__conversation-board__message__bubble"> <span>Winamp's still an essential.</span></div>
//             </div>
//             <div class="chat__conversation-board__message__options">
//               <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
//                 <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="10"></circle>
//                   <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
//                   <line x1="9" y1="9" x2="9.01" y2="9"></line>
//                   <line x1="15" y1="9" x2="15.01" y2="9"></line>
//                 </svg>
//               </button>
//               <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
//                 <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
//                   <circle cx="12" cy="12" r="1"></circle>
//                   <circle cx="19" cy="12" r="1"></circle>
//                   <circle cx="5" cy="12" r="1"></circle>
//                 </svg>
//               </button>
//             </div>
//           </div>
//         </div> -->"""

// write functions to create such html elements and append them to the chat box

function appendMessage(message, sender, buttons=false) {
    // create a div element
    const messageContainer = document.createElement("div")
    // add class to the div element
    messageContainer.classList.add("chat__conversation-board__message-container")
    if(sender.name === "You") {
        // add class to the div element
        messageContainer.classList.add("reversed")
    }
    // create a div element
    const person = document.createElement("div")
    // add class to the div element
    person.classList.add("chat__conversation-board__message__person")
    // create a div element
    const avatar = document.createElement("div")
    // add class to the div element
    avatar.classList.add("chat__conversation-board__message__person__avatar")
    // create an img element
    const img = document.createElement("img")
    // set the src attribute of the img element
    img.src = sender.avatar
    // set the alt attribute of the img element
    img.alt = sender.name
    // append the img element to the avatar div
    avatar.appendChild(img)
    // create a span element
    const nickname = document.createElement("span")
    // add class to the span element
    nickname.classList.add("chat__conversation-board__message__person__nickname")
    // set the text content of the span element
    nickname.textContent = sender.name
    // append the nickname span to the person div
    person.appendChild(avatar)
    person.appendChild(nickname)
    // create a div element
    const context = document.createElement("div")
    // add class to the div element
    context.classList.add("chat__conversation-board__message__context")
    // create a div element
    const bubble = document.createElement("div")
    // add class to the div element
    bubble.classList.add("chat__conversation-board__message__bubble")
    // create a span element
    const span = document.createElement("span")
    // set the text content of the span element
    span.innerHTML = message
    // append the span element to the bubble div
    bubble.appendChild(span)
    // append the bubble div to the context div
    context.appendChild(bubble)
    // append the person div to the message container div
    messageContainer.appendChild(person)
    // append the context div to the message container div
    messageContainer.appendChild(context)
    // create a div element
    const options = document.createElement("div")
    // add class to the div element
    options.classList.add("chat__conversation-board__message__options")
    // create a button element
    const emoji = document.createElement("button")
    
    // add class to the button element
    emoji.classList.add("btn-icon", "chat__conversation-board__message__option-button", "option-item", "emoji-button")
    // create an svg element
    const svg = document.createElement("svg")
    // set the class attribute of the svg element
    svg.setAttribute("class", "feather feather-smile sc-dnqmqq jxshSx")
    // set the xmlns attribute of the svg element
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    // set the width attribute of the svg element
    svg.setAttribute("width", "24")
    // set the height attribute of the svg element
    svg.setAttribute("height", "24")
    // set the viewBox attribute of the svg element
    svg.setAttribute("viewBox", "0 0 24 24")
    // set the fill attribute of the svg element
    svg.setAttribute("fill", "none")
    // set the stroke attribute of the svg element
    svg.setAttribute("stroke", "currentColor")
    // set the stroke-width attribute of the svg element
    svg.setAttribute("stroke-width", "2")
    // set the stroke-linecap attribute of the svg element
    svg.setAttribute("stroke-linecap", "round")
    // set the stroke-linejoin attribute of the svg element
    svg.setAttribute("stroke-linejoin", "round")
    // set the aria-hidden attribute of the svg element
    svg.setAttribute("aria-hidden", "true")
    // create a circle element
    const circle = document.createElement("circle")
    // set the cx attribute of the circle element
    circle.setAttribute("cx", "12")
    // set the cy attribute of the circle element
    circle.setAttribute("cy", "12")
    // set the r attribute of the circle element
    circle.setAttribute("r", "10")
    // create a path element
    const path = document.createElement("path")
    // set the d attribute of the path element
    path.setAttribute("d", "M8 14s1.5 2 4 2 4-2 4-2")
    // create a line element
    const line1 = document.createElement("line")
    // set the x1 attribute of the line element
    line1.setAttribute("x1", "9")
    // set the y1 attribute of the line element

    line1.setAttribute("y1", "9")
    // set the x2 attribute of the line element

    line1.setAttribute("x2", "9.01")
    // set the y2 attribute of the line element

    line1.setAttribute("y2", "9")
    // create a line element
    const line2 = document.createElement("line")
    // set the x1 attribute of the line element
    line2.setAttribute("x1", "15")
    // set the y1 attribute of the line element
    line2.setAttribute("y1", "9")
    // set the x2 attribute of the line element
    line2.setAttribute("x2", "15.01")
    // set the y2 attribute of the line element
    line2.setAttribute("y2", "9")
    // append the circle element to the svg element
    svg.appendChild(circle)

    // append the path element to the svg element
    svg.appendChild(path)
    // append the line1 element to the svg element
    svg.appendChild(line1)
    // append the line2 element to the svg element
    svg.appendChild(line2)
    // append the svg element to the emoji button
    emoji.appendChild(svg)
    // create a button element
    const more = document.createElement("button")
    // add class to the button element
    more.classList.add("btn-icon", "chat__conversation-board__message__option-button", "option-item", "more-button")
    // create an svg element
    const svg2 = document.createElement("svg")
    // set the class attribute of the svg element   

    svg2.setAttribute("class", "feather feather-more-horizontal sc-dnqmqq jxshSx")
    // set the xmlns attribute of the svg element
    svg2.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    // set the width attribute of the svg element
    svg2.setAttribute("width", "24")
    // set the height attribute of the svg element
    svg2.setAttribute("height", "24")
    // set the viewBox attribute of the svg element
    svg2.setAttribute("viewBox", "0 0 24 24")
    // set the fill attribute of the svg element
    svg2.setAttribute("fill", "none")
    // set the stroke attribute of the svg element
    svg2.setAttribute("stroke", "currentColor")
    // set the stroke-width attribute of the svg element
    svg2.setAttribute("stroke-width", "2")
    // set the stroke-linecap attribute of the svg element
    svg2.setAttribute("stroke-linecap", "round")
    // set the stroke-linejoin attribute of the svg element
    svg2.setAttribute("stroke-linejoin", "round")
    // set the aria-hidden attribute of the svg element
    svg2.setAttribute("aria-hidden", "true")
    // create a circle element
    const circle2 = document.createElement("circle")
    // set the cx attribute of the circle element
    circle2.setAttribute("cx", "12")
    // set the cy attribute of the circle element
    circle2.setAttribute("cy", "12")
    // set the r attribute of the circle element
    circle2.setAttribute("r", "1")
    // create a circle element
    const circle3 = document.createElement("circle")
    // set the cx attribute of the circle element
    circle3.setAttribute("cx", "19")
    // set the cy attribute of the circle element
    circle3.setAttribute("cy", "12")
    // set the r attribute of the circle element
    circle3.setAttribute("r", "1")
    // create a circle element
    const circle4 = document.createElement("circle")

    // set the cx attribute of the circle element
    circle4.setAttribute("cx", "5")
    // set the cy attribute of the circle element
    circle4.setAttribute("cy", "12")
    // set the r attribute of the circle element
    circle4.setAttribute("r", "1")
    // append the circle2 element to the svg2 element
    svg2.appendChild(circle2)
    // append the circle3 element to the svg2 element
    svg2.appendChild(circle3)
    // append the circle4 element to the svg2 element
    svg2.appendChild(circle4)
    // append the svg2 element to the more button
    more.appendChild(svg2)
    // append the emoji button to the options div
    options.appendChild(emoji)
    // append the more button to the options div
    options.appendChild(more)
    // append the options div to the message container div
    messageContainer.appendChild(options)
    if (buttons){
        const buttons = ["health care worker", "dentist", "first responder (e.g., teacher)"]
        const buttonContainer = document.createElement("div")
        buttonContainer.classList.add("chat__conversation-board__message__button-container")
        buttons.forEach(button => {
            const buttonElement = document.createElement("button")
            buttonElement.classList.add("btn", "btn-primary", "btn-sm", "btn-space")
            buttonElement.setAttribute("onclick", `startConvo("${button}")`)
            buttonElement.textContent = button
            buttonContainer.appendChild(buttonElement)
        })
        span.appendChild(buttonContainer)
    }
    // append the message container div to the chat box
    chatbox.append(messageContainer)
}

function startConvo(responderType){
    const message = `I am a ${responderType}`
    const sender = {
        name: "You",
        avatar: "https://th.bing.com/th/id/R.8e2c571ff125b3531705198a15d3103c?rik=gzhbzBpXBa%2bxMA&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fuser-png-icon-big-image-png-2240.png&ehk=VeWsrun%2fvDy5QDv2Z6Xm8XnIMXyeaz2fhR3AgxlvxAc%3d&risl=&pid=ImgRaw&r=0"
    }
    appendMessage(message, sender)
    getMessages(message)
}
// create a function to send messages
function sendMessage() {
    // get the message input element
    const messageInput = document.getElementById("message-input")
    // get the value of the message input element
    const message = messageInput.value
    // create a message object
    const messageObj = {
        message: message,
        sender: {
            name: "You",
            avatar: "https://th.bing.com/th/id/R.8e2c571ff125b3531705198a15d3103c?rik=gzhbzBpXBa%2bxMA&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fuser-png-icon-big-image-png-2240.png&ehk=VeWsrun%2fvDy5QDv2Z6Xm8XnIMXyeaz2fhR3AgxlvxAc%3d&risl=&pid=ImgRaw&r=0"
        }
    }
    // append the message to the chat box
    appendMessage(messageObj.message, messageObj.sender)
    // clear the message input
    messageInput.value = ""
    getMessages(message)
}

function scrollToBottom() {
    chatbox.scrollTop = chatbox.scrollHeight
}

function startMessage(){
    const message = "Hello, to start conversation, please select your responsder type?"
    const sender = {
        name: "Monika Figi",
        avatar: "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
    }
    appendMessage(message, sender, true)
    scrollToBottom()    
}

function addTypingIndicator() {
    const html_typer = `
	<svg viewBox="0 0 24 12" class="typing">
		<circle r="2" cx="4" cy="10" fill="currentColor">
			<animate attributeName="cy" values="10; 3; 10" dur="1.5s" begin="0s" repeatCount="indefinite" keyTimes="0;0.75;1" keySplines="1 0 1 0 1 0 1 0" calcMode="spline" />
		</circle>
		<circle r="2" cx="12" cy="10" fill="currentColor">
			<animate attributeName="cy" values="10; 3; 10" dur="1.5s" begin="-0.4s" repeatCount="indefinite" keyTimes="0;0.75;1" keySplines="1 0 1 0 1 0 1 0" calcMode="spline" />
		</circle>
		<circle r="2" cx="20" cy="10" fill="currentColor">
			<animate attributeName="cy" values="10; 3; 10" dur="1.5s" begin="-0.8s" repeatCount="indefinite" keyTimes="0;0.75;1" keySplines="1 0 1 0 1 0 1 0" calcMode="spline" />
		</circle>
	</svg>`

    appendMessage(html_typer, {name: "Monika Figi", avatar: "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"})
}

function removeTypingIndicator(){
    chatbox.removeChild(chatbox.lastChild)
}

// create a function to retrieve messages from the server
function getMessages(question) {
    // const messageObj = {}
    // create a message object
    // const messageObj = {
    //     message: "Hello lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nibh elementum imperdiet.",
    //     sender: {
    //         name: "Monika Figi",
    //         avatar: "https://th.bing.com/th/id/R.78399594cd4ce07c0246b0413c95f7bf?rik=Nwo0AAuaJO%2fPEQ&pid=ImgRaw&r=0"
    //     }
    // }
    // addTypingIndicator()
    // scrollToBottom()
    // // append the message to the chat box
    // setTimeout(() => {
        //     removeTypingIndicator()
        //     appendMessage(messageObj.message, messageObj.sender)
        //     scrollToBottom()
        // }, 3000);
        // fetch the messages from the server
        
        addTypingIndicator()
        scrollToBottom()
        fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
        body: JSON.stringify({'question': question})
    })
    .then(response => response.json())
    .then(data => {
        removeTypingIndicator()
        console.log(data);
        // loop through the messages
        var citation_box = document.getElementById("citations-box")
        let citationExists = false;
        citation_box.childNodes.forEach(child => {
            if (child.textContent === data.source) {
            citationExists = true;
            }
        });
        if (!citationExists && data.source) {
            var h6 = document.createElement("h6")
            h6.appendChild(document.createTextNode(data.source))
            citation_box.appendChild(h6)
            citation_box.appendChild(document.createElement("hr"))
        }
        // check in citation box each element, if the vale of any element is similar to the data.source, then don't append it


        console.log(data.source)
        // if (data.source){
        //     var h6 = document.createElement("h6")
        //     h6.appendChild(document.createTextNode(data.source))
        //     citation_box.appendChild(h6)
        // }
        formatted_response = marked.parse(data.response)
        appendMessage(formatted_response, data.sender)
        scrollToBottom()
    })
    .catch(error => {
        console.log(error);
    });
}   

// call the getMessages function
// getMessages()
// get the send button element
const sendButton = document.getElementById("send-button")
// add an event listener to the send button
sendButton.addEventListener("click", sendMessage)
// get the message input element
const messageInput = document.getElementById("message-input")
// add an event listener to the message input

messageInput.addEventListener("keypress", function(e) {
    // check if the enter key is pressed
    if (e.key === "Enter") {
        // call the sendMessage function
        sendMessage()
    }
}
)
const uploadContainer = document.getElementById('upload-container');
const fileInput = document.getElementById('fileInput');
const imageList = document.getElementById('imageList');


// Handle click to open file dialog
uploadContainer.addEventListener('click', () => {
    fileInput.click();
});

// Handle drag and drop events
uploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', () => {
    uploadContainer.classList.remove('dragover');
});

uploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadContainer.classList.remove('dragover');

    const files = e.dataTransfer.files;
    handleFiles(files);
});

// Handle file selection through dialog
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    handleFiles(files);
});

function handleFiles(files) {
  for (let file of files) {
      const reader = new FileReader();
      reader.onload = function(e) {
        console.log(e)
          const imageUrl = e.target.result;
          const imageItem = document.createElement('div');
          imageItem.setAttribute("onclick", "getImage(this)");
          imageItem.className = 'image-item';
          imageItem.innerHTML = `
              <h6 class="nameHandle" title="${file.name}">${file.name}</h6>
              <img src="${imageUrl}" alt="${file.name}" title="${file.name}" width="100%">
          `;
          // imageItem.innerHTML = `
          //     <h6>${file.name}</h6>
          //     <img src="${imageUrl}" alt="${file.name}" width="100%">
          // `;
          // imageList.appendChild(imageItem);
          // i want to sort the images in the order they are uploaded
          imageList.insertBefore(imageItem, imageList.firstChild);

      

        }
        reader.readAsDataURL(file);
}
}
startMessage()


function getImage(e){
    const image = e.getElementsByTagName("img")[0].src
    var image_html = `<img src="${image}" alt="image" width="50%">`
    appendMessage(image_html, {name: "You", avatar: "https://th.bing.com/th/id/R.8e2c571ff125b3531705198a15d3103c?rik=gzhbzBpXBa%2bxMA&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fuser-png-icon-big-image-png-2240.png&ehk=VeWsrun%2fvDy5QDv2Z6Xm8XnIMXyeaz2fhR3AgxlvxAc%3d&risl=&pid=ImgRaw&r=0"})
    addTypingIndicator()
    scrollToBottom()
    fetch("/get_image_description", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({'question': `Last message of bot is "Thank you for the information provided so far. To proceed with the appropriate guidance, could you please describe the severity of Hameed's dental injury? Specifically, has he lost consciousness or sustained other serious injuries that may require immediate medical attention? Additionally, any details about the condition of the affected tooth (e.g., chipped, knocked out, loose) would be helpful.

Could you please provide an image of Hameed's dental injury to help me better understand the situation?"`, 'image': image})
    })
    .then(response => response.json())
    .then(data => {
        removeTypingIndicator()
        console.log(data);
        // loop through the messages
        formatted_response = marked.parse(data.response)
        appendMessage(formatted_response, data.sender)
        scrollToBottom()
    })
    .catch(error => {
        console.log(error);
    });
    
    // getMessages(message)
}