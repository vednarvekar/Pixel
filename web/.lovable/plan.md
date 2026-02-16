

# Pixel — AI vs Real Image Detection App

## Overview
A sleek, dark-themed SPA with glassmorphism effects where users authenticate, upload images, and get AI-powered analysis of whether an image is real or AI-generated. The app connects to an existing Express/PyTorch backend at localhost (configurable) and uses Supabase for auth and scan history.

---

## Page 1: Landing Page
- Dark gradient background with subtle animated particles or grid pattern
- Hero section with the "Pixel" logo/brand name, tagline like *"Detect what's real in a world of fakes"*
- Brief animated feature highlights (how it works in 3 steps)
- CTA buttons: "Get Started" → signup, "Login" → login
- Glassmorphism card overlays with frosted blur effects

## Page 2: Auth Pages (Login & Signup)
- Full-screen split or centered card layout with dark background
- Glassmorphism card for the form
- Email/password auth via Supabase
- Google sign-in option
- Smooth fade-in animations on form elements
- Password reset flow included

## Page 3: Dashboard (Main App — Post Login)
- Top navbar with Pixel branding, user avatar, and logout
- **Drag & Drop Dropzone** at center:
  - Glassmorphism container with dashed border
  - On hover/drag: scale-up animation + glowing border effect
  - Accepts image files, shows instant preview after selection
  - Upload button with micro-interaction (scale on tap, lift on hover)

## Page 4: Scanning State
- After upload, smooth scroll down to results area
- **Scanning animation**: a glowing horizontal line moves up and down over the uploaded image preview
- **Skeleton loaders** below mimicking the layout of results (gauge, breakdown cards, verdict) — shimmering placeholders
- Feels like the AI is actively "analyzing" the image

## Page 5: Results Display
- **Verdict Score Gauge**: Large radial/circular progress gauge showing the final AI score
  - Color-coded: Green (<40% — likely real), Yellow (40-60% — uncertain), Light Red (60-75% — likely AI), Red (>75% — AI generated)
- **Verdict label**: Bold text like "Likely Real", "Uncertain", "Likely AI Generated", "AI Generated"
- **Score Breakdown Section**: Three glassmorphism cards showing:
  - Model Score (ResNet-18 prediction)
  - Metadata Score (EXIF analysis)
  - Web Score (reverse image search)
  - Each with its own small progress bar
- **Uploaded image** displayed on the left side below the gauge
- **"AI Reasoning" box**: Fades in with a typing/typewriter effect explaining the verdict — makes it feel like the AI is "thinking aloud"
- Button to "Scan Another Image" that scrolls back up

## Page 6: Scan History
- Accessible from the navbar
- Grid/list of past scans stored in Supabase
- Each card shows: thumbnail, date, final score with color indicator, verdict
- Click to expand and see full breakdown again
- Empty state with illustration when no scans yet

---

## Backend Integration
- Configurable API base URL (defaults to localhost:3000)
- POST to `/api/images/scan` with image as multipart form data
- Parse response: `final_score`, `verdict`, `breakdown` (model, metadata, web)

## Database (Supabase)
- **profiles** table: user profile info linked to auth
- **scans** table: stores scan history per user (image URL via Supabase storage, scores, verdict, timestamp)
- RLS policies so users only see their own scans

## Design System
- Dark base with deep navy/charcoal gradients
- Glassmorphism cards: `backdrop-blur`, semi-transparent backgrounds, subtle borders
- Accent color: Electric blue or cyan for highlights and glows
- Framer Motion for all micro-interactions (hover lifts, tap scales, fade-ins, scroll animations)
- Clean typography with Inter or similar modern font
- Responsive design for mobile and desktop

