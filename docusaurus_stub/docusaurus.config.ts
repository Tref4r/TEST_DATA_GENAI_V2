import { themes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';

const config: Config = {
  title: 'My Documentation',
  tagline: 'Technical Docs with Charts and Diagrams',
  url: 'https://your-site.com',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          remarkPlugins: [require('remark-math')],
          rehypePlugins: [require('rehype-katex')],
          exclude: [
            '**/note_conversion.md', // Matches note_conversion.md in any subdirectory
            '**/*Note0*.md',
            '**/*Note1*.md',
            '**/*DEP_*.md',
            // Add more patterns if needed, e.g., '**/*_draft.md'
          ],
          // Settings for autogenerating sidebar
          // This will be processed by our custom theme component
          sidebarItemsGenerator: async function ({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            return sortSidebarItems(sidebarItems);
          },
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  themes: ['@docusaurus/theme-mermaid'],
  themeConfig: {
    colorMode: {
      defaultMode: 'dark', // Set dark mode as default
    },
    tableOfContents: {
      minHeadingLevel: 2, // Start with # (level 1)
      maxHeadingLevel: 4, // End with #### (level 6)
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
    },
    mermaid: {
      theme: { light: 'default', dark: 'dark' },
    },
  },
  markdown: {
    mermaid: true,
  },
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  plugins: [
    function customWebpackPlugin() {
      return {
        name: 'custom-webpack-plugin',
        configureWebpack() {
          return {
            module: {
              rules: [
                {
                  test: /\.ya?ml$/,
                  use: 'yaml-loader',
                },
              ],
            },
          };
        },
      };
    },
    function tailwindPlugin() {
      return {
        name: 'tailwind-plugin',
        configurePostCss(postcssOptions) {
          // Append postcss-import and tailwindcss to the PostCSS config
      postcssOptions.plugins.push(
        require('@tailwindcss/postcss'),
        require('autoprefixer')
      );
          return postcssOptions;
        },
      };
    },
  ],
};

/**
 * Sort sidebar items to make folders (categories) appear before files
 */
function sortSidebarItems(items) {
  if (!Array.isArray(items)) return items;
  
  // Separate categories and docs
  const categories = items.filter(item => item.type === 'category');
  const docs = items.filter(item => item.type !== 'category');
  
  // Sort categories recursively
  categories.forEach(category => {
    if (category.items) {
      category.items = sortSidebarItems(category.items);
    }
  });
  
  // Return categories first, then docs
  return [...categories, ...docs];
}

export default config;
