# "Her" Movie-Inspired Design Palette

## Color Palette

Based on the movie "Her" (2013) directed by Spike Jonze, the film's production design features a warm, soft, and intimate color palette that reflects the emotional journey of the protagonist and the futuristic yet nostalgic aesthetic.

### Primary Colors (6-8 Key Colors)

1. **Coral Red** - `#f25c54`
   - Primary accent color, warm and inviting
   - Used for UI elements, buttons, and emotional highlights

2. **Soft Peach** - `#ffad9e`
   - Secondary warm tone
   - Perfect for hover states and gentle backgrounds

3. **Blush Pink** - `#ffceb6`
   - Tertiary warm tone
   - Ideal for subtle backgrounds and card elements

4. **Warm Orange** - `#ffd38d`
   - Complementary warm accent
   - Great for call-to-action elements and highlights

5. **Muted Teal** - `#4d6875`
   - Cool balance to warm tones
   - Perfect for text, borders, and secondary elements

6. **Midnight Blue** - `#2e3440`
   - Deep, sophisticated dark tone
   - Ideal for body text and dark UI elements

7. **Cream White** - `#faf8f1`
   - Soft, warm white
   - Primary background color

8. **Dusty Rose** - `#e8a598`
   - Additional warm tone
   - Great for subtle accents and dividers

## Typography

### Primary Font (Headings): Playfair Display
- **Google Font**: `Playfair Display`
- **Weight**: 400, 700
- **Style**: Serif, elegant, sophisticated
- **Usage**: Headlines, titles, featured text
- **Sizes**:
  - H1: 48px / Line height: 1.2
  - H2: 36px / Line height: 1.3
  - H3: 28px / Line height: 1.4
  - H4: 24px / Line height: 1.4

### Secondary Font (Body): Nunito Sans
- **Google Font**: `Nunito Sans`
- **Weight**: 300, 400, 600
- **Style**: Sans-serif, modern, readable
- **Usage**: Body text, navigation, UI elements
- **Sizes**:
  - Body: 16px / Line height: 1.6
  - Small: 14px / Line height: 1.5
  - Caption: 12px / Line height: 1.4

## Design Principles

### Color Usage Guidelines
- **Primary backgrounds**: Cream White (#faf8f1)
- **Text hierarchy**: Midnight Blue (#2e3440) for primary text, Muted Teal (#4d6875) for secondary
- **Accent elements**: Coral Red (#f25c54) for primary actions, Soft Peach (#ffad9e) for secondary
- **Warm highlights**: Warm Orange (#ffd38d) and Dusty Rose (#e8a598) for subtle emphasis

### Typography Pairing
- **Contrast**: Playfair Display (serif) provides elegant contrast to Nunito Sans (sans-serif)
- **Hierarchy**: Serif for emotional impact and attention, sans-serif for clarity and readability
- **Readability**: Nunito Sans ensures excellent readability across all devices

### Emotional Design Elements
- **Warmth**: Dominated by warm tones that evoke intimacy and human connection
- **Sophistication**: Deep blues and elegant typography create a refined aesthetic
- **Accessibility**: Color contrast ratios ensure readability and inclusivity
- **Modernity**: Clean sans-serif paired with classic serif creates timeless appeal

## Implementation

### CSS Variables
```css
:root {
  /* Colors */
  --color-coral-red: #f25c54;
  --color-soft-peach: #ffad9e;
  --color-blush-pink: #ffceb6;
  --color-warm-orange: #ffd38d;
  --color-muted-teal: #4d6875;
  --color-midnight-blue: #2e3440;
  --color-cream-white: #faf8f1;
  --color-dusty-rose: #e8a598;
  
  /* Typography */
  --font-heading: 'Playfair Display', serif;
  --font-body: 'Nunito Sans', sans-serif;
  
  /* Font Sizes */
  --font-size-h1: 48px;
  --font-size-h2: 36px;
  --font-size-h3: 28px;
  --font-size-h4: 24px;
  --font-size-body: 16px;
  --font-size-small: 14px;
  --font-size-caption: 12px;
  
  /* Line Heights */
  --line-height-tight: 1.2;
  --line-height-normal: 1.4;
  --line-height-relaxed: 1.6;
}
```

### Google Fonts Import
```css
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Nunito+Sans:wght@300;400;600&display=swap');
```

This palette captures the intimate, warm, and slightly futuristic aesthetic of "Her" while maintaining excellent usability and accessibility standards for modern web design.
