import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'
import svelte from 'eslint-plugin-svelte'
import prettier from 'eslint-config-prettier'

// ESLint flat config. Order matters: language rules first, then Svelte, then
// prettier last so it can turn off any formatting-related lint rules.
export default tseslint.config(
  {
    ignores: ['dist/', 'node_modules/', 'playwright-report/', 'test-results/', '.svelte-kit/'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...svelte.configs['flat/recommended'],
  {
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
  },
  {
    files: ['**/*.svelte'],
    languageOptions: {
      parserOptions: {
        // Let the Svelte parser hand <script lang="ts"> off to the TS parser.
        parser: tseslint.parser,
      },
    },
  },
  prettier,
  ...svelte.configs['flat/prettier'],
)
