function showNotif(message) {
  const popup = document.getElementById("popup");
  if (!popup) { alert(message); return; }
  popup.textContent = message;
  popup.style.display = "block";
  setTimeout(() => { popup.style.display = "none"; }, 2500);
}
