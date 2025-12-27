# Market Scanner API - GitHub Pages

Static JSON API for Market Scanner Flutter App.  
Data is automatically updated every 5 minutes via GitHub Actions.

## API Endpoints

After setup, your API will be available at:
```
https://<your-username>.github.io/market-scanner-api/data/
```

### Available Endpoints:

| Endpoint | Description |
|----------|-------------|
| `/data/health.json` | Health check & last update time |
| `/data/crypto_signals.json` | Crypto trading signals |
| `/data/thai_signals.json` | Thai stock signals |
| `/data/all_signals.json` | Combined signals (for polling) |

## Setup Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `market-scanner-api`
3. Make it **Public** (required for GitHub Pages)
4. Click "Create repository"

### Step 2: Upload Files

Upload the entire `github_api` folder contents to your new repo:
- `.github/workflows/update_data.yml`
- `scripts/fetch_data.py`
- `data/*.json`
- `README.md`

Or use git:
```bash
cd github_api
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/market-scanner-api.git
git push -u origin main
```

### Step 3: Enable GitHub Pages

1. Go to your repo → Settings → Pages
2. Source: **Deploy from a branch**
3. Branch: **main** / **/ (root)**
4. Click Save

### Step 4: Enable GitHub Actions

1. Go to your repo → Actions
2. Click "I understand my workflows, go ahead and enable them"
3. The workflow will run automatically every 5 minutes

### Step 5: Update Flutter App

Update your Flutter app's `api_service.dart`:

```dart
class ApiService {
  // Change this to your GitHub Pages URL
  static const String baseUrl = 'https://<your-username>.github.io/market-scanner-api/data';
  
  Future<List<Signal>> getCryptoSignals() async {
    final response = await http.get(Uri.parse('$baseUrl/crypto_signals.json'));
    // ... parse JSON
  }
}
```

## How It Works

1. **GitHub Actions** runs every 5 minutes
2. **Python script** fetches data from Binance & Yahoo Finance
3. **JSON files** are updated in the `data/` folder
4. **GitHub Pages** serves the JSON files as static API

## Advantages

✅ **100% Free** - GitHub Pages is free  
✅ **No Rate Limits** - Static files, no server processing  
✅ **Always Available** - GitHub has 99.9% uptime  
✅ **Fast CDN** - GitHub Pages uses Fastly CDN  
✅ **No Server Maintenance** - Zero ops required  

## Data Update Frequency

- **Crypto data**: Updated every 5 minutes from Binance
- **Thai stocks**: Updated every 5 minutes from Yahoo Finance
- **RSI/MACD**: Calculated with each update

## Troubleshooting

### Actions not running?
1. Go to Actions tab
2. Click on the workflow
3. Check for errors in the logs

### Data not updating?
1. Check if Actions are enabled
2. Look for failed workflow runs
3. Verify the script doesn't have errors

### Pages not working?
1. Check Settings → Pages
2. Ensure branch is set to `main`
3. Wait 1-2 minutes for deployment

## License

MIT License
