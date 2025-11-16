# Playground UI

This directory will contain the web-based tokenization playground.

## Status

**Current:** Planning and API specification phase

**Future:** Implementation of web UI for interactive tokenizer comparison

## Planned Features

- Interactive text input with language selection
- Multi-tokenizer comparison
- Visual token boundary display
- Metrics visualization
- Failure case browser
- Scorecard generation

## API Specification

See `API_SPEC.md` for detailed API endpoint specifications.

## Implementation Plan

1. **Phase 1: Backend API**
   - FastAPI or Flask backend
   - Implement API endpoints
   - Connect to existing Python modules

2. **Phase 2: Frontend UI**
   - Next.js or vanilla JS frontend
   - Token visualization components
   - Metrics display

3. **Phase 3: Integration**
   - Connect frontend to backend
   - Add real-time updates
   - Polish UI/UX

## Quick Start (Future)

```bash
# Install dependencies
cd playground
npm install  # or pip install -r requirements.txt

# Start backend
python api/server.py

# Start frontend (if separate)
npm run dev
```

## See Also

- `docs/30-playground-ui-spec.md` - Detailed UI specification
- `playground/API_SPEC.md` - API endpoint documentation
