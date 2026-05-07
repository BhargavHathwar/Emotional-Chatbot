function sendMessage(){

let input = document.getElementById("message");
let msg = input.value;

if(msg.trim()===""){
return;
}

let chat = document.getElementById("chatbox");
let typing = document.getElementById("typing");

/* user bubble */

chat.innerHTML +=
'<div class="message user">'+msg+'</div>';

chat.scrollTop = chat.scrollHeight;

typing.style.display="block";

/* send to backend */

fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({message:msg})

})

.then(res=>res.json())

.then(data=>{

setTimeout(()=>{

typing.style.display="none";

chat.innerHTML +=
'<div class="message bot">'+data.reply+'</div>';

chat.scrollTop = chat.scrollHeight;

},800);

});

input.value="";

}

/* press enter */

document.getElementById("message").addEventListener("keypress",function(e){

if(e.key==="Enter"){

sendMessage();

}

});