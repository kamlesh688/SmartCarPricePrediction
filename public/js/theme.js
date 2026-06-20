const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
if (!localStorage.getItem('site-theme')) {
  document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
}
