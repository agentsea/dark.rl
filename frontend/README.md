# Dark.RL Frontend

A modern React TypeScript web interface for the Dark.RL reinforcement learning platform.

## Features

- **Model Interaction**: Send prompts to your trained models and view responses
- **System Status**: Monitor active models, training jobs, and system uptime
- **Modern UI**: Clean, responsive design with custom CSS styling
- **Real-time Updates**: Live feedback on model processing

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Custom CSS** for styling (simplified approach)
- **Responsive Design** with mobile-first approach

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and go to `http://localhost:5173`

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run linter

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components (shadcn/ui components)
│   ├── lib/           # Utility functions
│   ├── App.tsx        # Main application component
│   ├── main.tsx       # Application entry point
│   └── index.css      # Global styles
├── public/            # Static assets
└── package.json       # Dependencies and scripts
```

## Features Overview

### Model Interaction
- Submit text prompts to your models
- Real-time processing indicators
- Response display with proper formatting

### System Monitoring
- Real-time system status
- Active model tracking
- Training job statistics
- Uptime monitoring

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized
- Clean, modern interface

## API Integration

The frontend is designed to integrate with the Dark.RL backend API. Currently, it includes mock responses for demonstration. To connect to your actual backend:

1. Update the API endpoints in the component
2. Implement proper error handling
3. Add authentication if required

## Customization

### Styling
The application uses custom CSS for styling. You can modify `src/index.css` to customize:
- Colors and themes
- Typography
- Layout and spacing
- Component styles

### Components
Add new components in the `src/components` directory. The project supports:
- Functional components with hooks
- TypeScript interfaces
- Modular component structure

## Building for Production

1. Build the application:
   ```bash
   npm run build
   ```

2. The built files will be in the `dist/` directory

3. Serve the built files using any static file server

## Deployment

The built application is a static site that can be deployed to:
- Netlify
- Vercel
- GitHub Pages
- AWS S3
- Any static hosting provider

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Test your changes thoroughly
4. Keep components modular and reusable

## License

This project is part of the Dark.RL platform and follows the same licensing terms.
