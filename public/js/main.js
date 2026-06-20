const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('site-theme', next);
  });
}
const savedTheme = localStorage.getItem('site-theme');
if (savedTheme) {
  document.documentElement.setAttribute('data-theme', savedTheme);
}
