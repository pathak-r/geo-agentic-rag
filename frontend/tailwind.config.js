/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { navy: "#1E3A5F", accent: "#4A90D9" },
      },
    },
  },
  plugins: [],
};
