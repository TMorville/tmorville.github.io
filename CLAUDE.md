# Technical Documentation

## Project Type
Jekyll-based GitHub Pages personal website using Minimal Mistakes theme.

## Key Technical Details

### Framework & Theme
- **Generator**: Jekyll static site generator
- **Theme**: Minimal Mistakes v4.9.0
- **Skin**: "sunrise" 
- **Hosting**: GitHub Pages

### Structure
```
├── _posts/              # Blog posts (markdown)
├── _posts_wip/          # Draft posts
├── _pages/              # Static pages (about, gallery, resume)
├── _config.yml          # Main Jekyll configuration
├── _data/               # YAML data files (navigation, UI text)
├── _includes/           # Reusable HTML components
├── _layouts/            # Page templates
├── _sass/               # Theme styling
├── assets/              # Static assets (images, CSS, JS, PDFs)
└── docs/                # Theme documentation (excluded from build)
```

### Configuration
- **Permalink**: `/:categories/:title/`
- **Pagination**: 5 posts per page
- **Markdown**: Kramdown with GFM input
- **Highlighter**: Rouge
- **Compression**: HTML compression enabled

### Plugins
- jekyll-paginate
- jekyll-sitemap  
- jekyll-gist
- jekyll-feed
- jemoji

### Content Types
- **Posts**: Technical blog posts (2016-2020)
- **Pages**: About, gallery, resume
- **Assets**: Research PDFs, data visualizations, art gallery

### Build Process
- **JavaScript**: Uses npm scripts for minification/bundling
- **CSS**: Sass compilation via Jekyll
- **Deployment**: Automatic via GitHub Pages

### Author Configuration
- Location: Copenhagen
- Email: tobiasmorville@gmail.com
- GitHub: TMorville
- LinkedIn: tobias-morville-24b97312

### System Requirements
- **Ruby**: 3.1+ (installed via Homebrew)
- **Node.js**: 16+ 
- **Bundler**: 2.0+

### Setup Instructions
```bash
# Install Ruby via Homebrew (if not done)
brew install ruby

# Add Ruby to PATH (add to ~/.zshrc)
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/3.4.0/bin:$PATH"

# Install dependencies
gem install bundler
bundle update
npm update
```

### Development Commands
```bash
# Build site
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/3.4.0/bin:$PATH"
bundle exec jekyll build

# Local development server
bundle exec jekyll serve
# Site available at: http://localhost:4000

# JS build process  
npm run build:js
npm run watch:js
```

### Updated Versions
- **Jekyll**: 4.4.1 (was 3.6)
- **Minimal Mistakes**: 4.27.1 (was 4.9.0)
- **Ruby**: 3.4.4 (was 2.6.10)
- **Node**: 16+ (was 0.10.0)