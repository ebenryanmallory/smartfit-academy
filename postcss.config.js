export default {
  plugins: {
    '@tailwindcss/postcss': {
      config: './tailwind.config.js'
    },
    'autoprefixer': {
      flexbox: 'no-2009'
    },
    'postcss-preset-env': {
      stage: 3,
      features: {
        'nesting-rules': true
      }
    }
  }
}