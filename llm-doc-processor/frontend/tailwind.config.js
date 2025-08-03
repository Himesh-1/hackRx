/** @type {import('tailwindcss').Config} */
export const content = [
  "./src/**/*.{js,jsx,ts,tsx}",
  "./public/index.html"
];
export const darkMode = 'class';
export const theme = {
  extend: {
    animation: {
      'in': 'fadeIn 0.5s ease-in-out',
      'slide-in-from-bottom-4': 'slideInFromBottom 0.5s ease-out',
    },
    keyframes: {
      fadeIn: {
        '0%': { opacity: '0' },
        '100%': { opacity: '1' },
      },
      slideInFromBottom: {
        '0%': { transform: 'translateY(16px)', opacity: '0' },
        '100%': { transform: 'translateY(0)', opacity: '1' },
      },
    },
  },
};
export const plugins = [];