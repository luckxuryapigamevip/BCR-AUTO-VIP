const axios = require('axios');
const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');

// ======================
// CẤU HÌNH
// ======================
const BASE = "https://aibcr.me";
const LOGIN_URL = `${BASE}/login`;
const LOBBY_URL = `${BASE}/ae/lobby`;
const GETNEWRESULT_URL = `${BASE}/baccarat/getnewresult`;

const USERNAME = "Hoang2285";
const PASSWORD = "hoang2010";

const agent = new https.Agent({ rejectUnauthorized: false });
let cookieJar = '';
let baccaratData = [];
let lastUpdate = null;

// ============================================================
// BCR – ADAPTIVE MULTI-PATTERN LEARNING (AML)
// Dự đoán từ phiên đầu — vừa đoán vừa học song song
// ============================================================
const MIN_PATTERN_LEN = 4;
const MAX_PATTERN_LEN = 12;
const DECAY = 0.95;
const BOOST_CORRECT = 1.0;
const PENALTY_WRONG = -1.5;
const DECAY_EVERY = 50;

const RATIO_MIN = 55.0;
const RATIO_MAX = 89.0;

const AML_STATE_FILE = path.join(__dirname, 'aml_state.json');

class PatternBook {
    constructor() {
        this.book = new Map();
        this.history = [];
        this.lastPrediction = null;
        this.sessionsSinceDecay = 0;
        this.stats = { total: 0, correct: 0, wrong: 0 };
    }

    _newRecord() {
        return {
            count: 0,
            predict_B: 0, predict_P: 0, predict_T: 0,
            wrong_B: 0, wrong_P: 0, wrong_T: 0,
            weight: 1.0
        };
    }

    _get(pat) {
        let rec = this.book.get(pat);
        if (!rec) {
            rec = this._newRecord();
            this.book.set(pat, rec);
        }
        return rec;
    }

    // BƯỚC 1+2: học mẫu mới + settle dự đoán trước
    pushResult(r) {
        if (!['B', 'P', 'T'].includes(r)) return;

        this.history.push(r);
        const endIdx = this.history.length;

        for (let L = MIN_PATTERN_LEN; L <= MAX_PATTERN_LEN; L++) {
            if (endIdx - L < 1) break;
            const patArr = this.history.slice(endIdx - L - 1, endIdx - 1);
            const pat = patArr.join('');
            const rec = this._get(pat);
            rec.count += 1;
            rec['predict_' + r] += 1;
        }

        if (this.lastPrediction && this.lastPrediction.history_len === endIdx - 1) {
            this._settle(this.lastPrediction, r);
            this.lastPrediction = null;
        }

        this.sessionsSinceDecay++;
        if (this.sessionsSinceDecay >= DECAY_EVERY) {
            for (const rec of this.book.values()) rec.weight *= DECAY;
            this.sessionsSinceDecay = 0;
        }
    }

    _settle(last, actual) {
        const ok = (actual === last.pred);
        this.stats.total++;
        if (ok) this.stats.correct++;
        else this.stats.wrong++;

        for (const entry of last.pattern_set) {
            const rec = this.book.get(entry.pat);
            if (!rec) continue;
            if (ok) {
                rec.weight += BOOST_CORRECT;
            } else {
                rec.weight += PENALTY_WRONG;
                rec['wrong_' + last.pred] += 1;
            }
        }
    }

    // ===== DỰ ĐOÁN — LUÔN TRẢ VỀ 1 HƯỚNG =====
    predict() {
        const H = this.history;
        const n = H.length;

        // Trường hợp 0 phiên — mặc định B
        if (n === 0) {
            this.lastPrediction = { pattern_set: [], pred: 'B', history_len: 0 };
            return {
                pred: 'B',
                scores: { B: 0.34, P: 0.33, T: 0.33 },
                candidates: 0,
                confidence: 0.34,
                source: 'baseline_empty'
            };
        }

        // Thu thập candidates từ mọi độ dài khả dụng (kể cả < MIN_PATTERN_LEN)
        const candidates = [];
        for (let L = 1; L <= MAX_PATTERN_LEN; L++) {
            if (n < L) break;
            const pat = H.slice(H.length - L).join('');
            const rec = this.book.get(pat);
            if (!rec) continue;
            const total = rec.predict_B + rec.predict_P + rec.predict_T;
            if (total === 0) continue;
            candidates.push({
                pat,
                L,
                score_B: (rec.predict_B - rec.wrong_B) / total,
                score_P: (rec.predict_P - rec.wrong_P) / total,
                score_T: (rec.predict_T - rec.wrong_T) / total,
                weight: rec.weight
            });
        }

        // Chưa có pattern khớp → fallback thống kê
        if (candidates.length === 0) {
            return this._fallbackPredict(H);
        }

        let W_B = 0, W_P = 0, W_T = 0;
        for (const c of candidates) {
            const w = Math.max(0.0001, c.weight);
            W_B += c.score_B * w;
            W_P += c.score_P * w;
            W_T += c.score_T * w;
        }

        const scores = { B: W_B, P: W_P, T: W_T };

        // Toàn bộ score = 0 → fallback
        if (W_B + W_P + W_T < 1e-9) {
            return this._fallbackPredict(H);
        }

        let pred = 'B';
        if (scores.P > scores[pred]) pred = 'P';
        if (scores.T > scores[pred]) pred = 'T';

        this.lastPrediction = {
            pattern_set: candidates,
            pred,
            history_len: n
        };

        const totalW = W_B + W_P + W_T;
        const conf = totalW > 0 ? scores[pred] / totalW : 0;

        return {
            pred,
            scores: {
                B: +W_B.toFixed(4),
                P: +W_P.toFixed(4),
                T: +W_T.toFixed(4)
            },
            candidates: candidates.length,
            confidence: +conf.toFixed(4),
            source: 'pattern_book'
        };
    }

    // ===== Fallback: tần suất + xu hướng gần + đối xứng =====
    _fallbackPredict(H) {
        const n = H.length;

        // 1) Tần suất toàn cục (Laplace smoothing)
        let cntB = 0, cntP = 0, cntT = 0;
        for (const ch of H) {
            if (ch === 'B') cntB++;
            else if (ch === 'P') cntP++;
            else if (ch === 'T') cntT++;
        }
        const freq = {
            B: (cntB + 1) / (n + 3),
            P: (cntP + 1) / (n + 3),
            T: (cntT + 1) / (n + 3)
        };

        // 2) Xu hướng 10 phiên gần nhất
        const recent = n >= 5 ? H.slice(-10) : H.slice(-n);
        let rB = 0, rP = 0, rT = 0;
        for (const ch of recent) {
            if (ch === 'B') rB++;
            else if (ch === 'P') rP++;
            else if (ch === 'T') rT++;
        }
        const m = recent.length;
        const recentFreq = {
            B: (rB + 0.5) / (m + 1.5),
            P: (rP + 0.5) / (m + 1.5),
            T: (rT + 0.5) / (m + 1.5)
        };

        // 3) Đối xứng: kết quả cuối B → nghiêng P, và ngược lại
        const last = n > 0 ? H[n - 1] : null;
        const symmetry = { B: 0, P: 0, T: 0 };
        if (last === 'B') { symmetry.P = 0.15; symmetry.B = -0.05; }
        else if (last === 'P') { symmetry.B = 0.15; symmetry.P = -0.05; }

        // Kết hợp
        const combined = {
            B: freq.B * 0.4 + recentFreq.B * 0.4 + symmetry.B * 0.2,
            P: freq.P * 0.4 + recentFreq.P * 0.4 + symmetry.P * 0.2,
            T: freq.T * 0.4 + recentFreq.T * 0.4 + symmetry.T * 0.2
        };

        let pred = 'B';
        if (combined.P > combined[pred]) pred = 'P';
        if (combined.T > combined[pred]) pred = 'T';

        const total = combined.B + combined.P + combined.T;
        const conf = total > 0 ? combined[pred] / total : 0.34;

        this.lastPrediction = {
            pattern_set: [],
            pred,
            history_len: n
        };

        return {
            pred,
            scores: {
                B: +combined.B.toFixed(4),
                P: +combined.P.toFixed(4),
                T: +combined.T.toFixed(4)
            },
            candidates: 0,
            confidence: +conf.toFixed(4),
            source: 'fallback_stats'
        };
    }

    toJSON() {
        return {
            history: this.history,
            book: Array.from(this.book.entries()),
            stats: this.stats,
            sessionsSinceDecay: this.sessionsSinceDecay
        };
    }

    static fromJSON(obj) {
        const pb = new PatternBook();
        pb.history = obj.history || [];
        pb.stats = obj.stats || { total: 0, correct: 0, wrong: 0 };
        pb.sessionsSinceDecay = obj.sessionsSinceDecay || 0;
        pb.book = new Map(obj.book || []);
        return pb;
    }
}

class AMLManager {
    constructor() {
        this.books = new Map(); // table -> { book, lastKnownResult }
    }

    sync(tableName, resultString) {
        if (!resultString) return;

        let entry = this.books.get(tableName);
        if (!entry) {
            entry = { book: new PatternBook(), lastKnownResult: '' };
            this.books.set(tableName, entry);
            for (const ch of resultString) entry.book.pushResult(ch);
            entry.lastKnownResult = resultString;
            entry.book.predict();
            return;
        }

        const prev = entry.lastKnownResult;
        if (resultString.length > prev.length) {
            if (resultString.startsWith(prev)) {
                const added = resultString.slice(prev.length);
                for (const ch of added) entry.book.pushResult(ch);
            } else {
                entry.book = new PatternBook();
                for (const ch of resultString) entry.book.pushResult(ch);
            }
            entry.lastKnownResult = resultString;
            entry.book.predict();
        } else if (resultString.length < prev.length) {
            entry.book = new PatternBook();
            for (const ch of resultString) entry.book.pushResult(ch);
            entry.lastKnownResult = resultString;
            entry.book.predict();
        }
    }

    // ===== TRẢ JSON ĐÚNG FORMAT — LUÔN CÓ DỰ ĐOÁN =====
    getPrediction(tableName) {
        const entry = this.books.get(tableName);
        if (!entry) return null;

        const book = entry.book;
        const H = book.history;
        const n = H.length;

        // Luôn gọi predict() — nó tự fallback nếu chưa đủ dữ liệu
        const p = book.predict();
        const pred = p.pred || 'B';

        // TÍNH TỈ LỆ
        const W_B = p.scores.B;
        const W_P = p.scores.P;
        const W_T = p.scores.T;
        const W_sum = W_B + W_P + W_T;
        const ratioByWeight = W_sum > 0 ? (p.scores[pred] / W_sum) : 0.34;

        // Tỷ lệ mẫu ủng hộ
        let totalSamples = 0;
        let correctSamples = 0;
        const patternSet = (book.lastPrediction && book.lastPrediction.pattern_set) || [];
        for (const c of patternSet) {
            const rec = book.book.get(c.pat);
            if (!rec) continue;
            totalSamples += rec.count;
            const wrongForPred = rec['wrong_' + pred] || 0;
            correctSamples += Math.max(0, rec['predict_' + pred] - wrongForPred);
        }
        const ratioBySamples = totalSamples > 0 ? (correctSamples / totalSamples) : ratioByWeight;

        // Kết hợp
        let finalRatio;
        if (totalSamples >= 10) {
            finalRatio = ratioByWeight * 0.4 + ratioBySamples * 0.6;
        } else if (totalSamples >= 3) {
            finalRatio = ratioByWeight * 0.7 + ratioBySamples * 0.3;
        } else {
            finalRatio = ratioByWeight;
        }

        let percent = finalRatio * 100;

        // KHÔNG BAO GIỜ NULL — nếu dưới ngưỡng, nâng lên baseline tuyến tính
        if (percent < RATIO_MIN) {
            percent = RATIO_MIN + (percent / 100) * 5;
        }
        if (percent > RATIO_MAX) percent = RATIO_MAX;

        // Cầu hiển thị
        let cau;
        if (patternSet.length > 0) {
            const longest = [...patternSet].sort((a, b) => b.L - a.L)[0];
            cau = longest.pat;
        } else {
            cau = H.slice(-MAX_PATTERN_LEN).join('');
        }
        if (!cau && n === 0) cau = '—';

        return {
            "bàn": tableName,
            "phiên": n,
            "cầu": cau,
            "dự đoán": pred,
            "tỉ lệ": percent.toFixed(1) + '%'
        };
    }

    getAllPredictions() {
        const out = [];
        for (const [tableName] of this.books) {
            const p = this.getPrediction(tableName);
            if (p) out.push(p);
        }
        return out;
    }

    saveToFile() {
        try {
            const dump = {};
            for (const [t, e] of this.books) {
                dump[t] = {
                    book: e.book.toJSON(),
                    lastKnownResult: e.lastKnownResult
                };
            }
            fs.writeFileSync(AML_STATE_FILE, JSON.stringify(dump));
            console.log(`[AML] Đã lưu state (${Object.keys(dump).length} bàn)`);
        } catch (err) {
            console.error('[AML] Lỗi lưu state:', err.message);
        }
    }

    loadFromFile() {
        try {
            if (!fs.existsSync(AML_STATE_FILE)) return false;
            const dump = JSON.parse(fs.readFileSync(AML_STATE_FILE, 'utf8'));
            for (const [t, obj] of Object.entries(dump)) {
                this.books.set(t, {
                    book: PatternBook.fromJSON(obj.book),
                    lastKnownResult: obj.lastKnownResult || ''
                });
            }
            console.log(`[AML] Đã khôi phục state (${Object.keys(dump).length} bàn)`);
            return true;
        } catch (err) {
            console.error('[AML] Lỗi load state:', err.message);
            return false;
        }
    }
}

const aml = new AMLManager();

function syncAMLFromData() {
    for (const item of baccaratData) {
        if (item.table && item.result) {
            aml.sync(item.table, item.result);
        }
    }
}

// ======================
// SESSION AXIOS
// ======================
const session = axios.create({
    baseURL: BASE,
    timeout: 30000,
    httpsAgent: agent,
    headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7'
    }
});

session.interceptors.request.use(config => {
    if (cookieJar) config.headers.Cookie = cookieJar;
    return config;
});

session.interceptors.response.use(res => {
    const setCookie = res.headers['set-cookie'];
    if (setCookie) {
        for (const cookie of setCookie) {
            const [name, value] = cookie.split(';')[0].split('=');
            if (cookieJar.includes(`${name}=`)) {
                cookieJar = cookieJar.replace(new RegExp(`${name}=[^;]+;?`), '');
            }
            cookieJar += `${name}=${value}; `;
        }
    }
    return res;
});

// ======================
// LẤY CSRF TOKEN
// ======================
function getCsrfToken(html) {
    const match = html.match(/<meta\s+name="csrf-token"\s+content="([^"]+)"/);
    return match ? match[1] : null;
}

// ======================
// ĐĂNG NHẬP
// ======================
async function login() {
    try {
        const getResp = await session.get(LOGIN_URL);
        const token = getCsrfToken(getResp.data);

        const formData = new URLSearchParams();
        formData.append('username', USERNAME);
        formData.append('password', PASSWORD);
        formData.append('_token', token);
        formData.append('action', 'Login');

        const headers = {
            'Referer': LOGIN_URL,
            'Origin': BASE,
            'Content-Type': 'application/x-www-form-urlencoded'
        };

        const loginResp = await session.post(LOGIN_URL, formData.toString(), { headers });
        return loginResp.status === 200;
    } catch (error) {
        console.error('Login error:', error.message);
        return false;
    }
}

// ======================
// VÀO LOBBY
// ======================
async function goToLobby() {
    try {
        await session.get(LOBBY_URL);
        return true;
    } catch (error) {
        console.error('Lobby error:', error.message);
        return false;
    }
}

// ======================
// LẤY KẾT QUẢ BACCARAT
// ======================
async function fetchBaccaratData() {
    try {
        let xsrfToken = '';
        const xsrfMatch = cookieJar.match(/XSRF-TOKEN=([^;]+)/);
        if (xsrfMatch) xsrfToken = decodeURIComponent(xsrfMatch[1]);

        const headers = {
            'Referer': LOBBY_URL,
            'Origin': BASE,
            'X-Requested-With': 'XMLHttpRequest',
            'X-XSRF-TOKEN': xsrfToken,
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
        };

        const formData = new URLSearchParams();
        formData.append('gameCode', 'ae');

        const resp = await session.post(GETNEWRESULT_URL, formData.toString(), { headers });

        if (resp.data && resp.data.data) {
            baccaratData = resp.data.data.map(item => {
                let resultStr = item.result;
                if (Array.isArray(resultStr)) {
                    resultStr = resultStr.map(r => (r && r.result) ? r.result : String(r)).join('');
                }
                return {
                    table: item.table_name,
                    result: resultStr || '',
                    shoeId: item.shoeId || '',
                    round: item.round || ''
                };
            });
            lastUpdate = new Date().toISOString();
            syncAMLFromData();
        }

        return baccaratData;
    } catch (error) {
        console.error('Fetch error:', error.message);
        return [];
    }
}

// ======================
// VÒNG LẶP TỰ ĐỘNG CẬP NHẬT
// ======================
async function autoUpdate() {
    while (true) {
        await fetchBaccaratData();
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}

// ======================
// KHỞI TẠO API SERVER
// ======================
const app = express();

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', '*');
    next();
});

// API lấy tất cả bàn
app.get('/api/baccarat', (req, res) => {
    res.json({
        success: true,
        data: baccaratData,
        lastUpdate: lastUpdate,
        total: baccaratData.length
    });
});

// API lấy theo bàn cụ thể
app.get('/api/baccarat/:table', (req, res) => {
    const tableName = req.params.table;
    const found = baccaratData.find(item => item.table === tableName);
    if (found) {
        res.json({ success: true, data: found });
    } else {
        res.json({ success: false, message: 'Không tìm thấy bàn ' + tableName });
    }
});

// API lấy kết quả mới nhất
app.get('/api/latest', (req, res) => {
    const latest = [...baccaratData].sort((a, b) => {
        const numA = parseInt(a.table) || 0;
        const numB = parseInt(b.table) || 0;
        return numB - numA;
    });
    res.json({ success: true, data: latest.slice(0, 10), lastUpdate: lastUpdate });
});

// ======================
// API AML — DỰ ĐOÁN (LUÔN CÓ KẾT QUẢ)
// ======================
app.get('/api/predict', (req, res) => {
    const all = aml.getAllPredictions();
    res.json({
        success: true,
        total: all.length,
        data: all,
        lastUpdate: lastUpdate
    });
});

app.get('/api/predict/:table', (req, res) => {
    const p = aml.getPrediction(req.params.table);
    if (!p) {
        return res.json({
            success: false,
            message: 'Bàn chưa có dữ liệu: ' + req.params.table
        });
    }
    res.json({ success: true, data: p, lastUpdate: lastUpdate });
});

app.get('/api/predict/:table/detail', (req, res) => {
    const entry = aml.books.get(req.params.table);
    if (!entry) return res.json({ success: false, message: 'Không có dữ liệu' });

    const book = entry.book;
    const patterns = [];
    for (const [pat, rec] of book.book.entries()) {
        if (rec.count >= 2) {
            patterns.push({
                pattern: pat,
                count: rec.count,
                predict: { B: rec.predict_B, P: rec.predict_P, T: rec.predict_T },
                wrong: { B: rec.wrong_B, P: rec.wrong_P, T: rec.wrong_T },
                weight: +rec.weight.toFixed(3)
            });
        }
    }
    patterns.sort((a, b) => b.weight - a.weight);

    res.json({
        success: true,
        table: req.params.table,
        prediction: aml.getPrediction(req.params.table),
        top_patterns: patterns.slice(0, 20),
        total_patterns: book.book.size,
        stats: book.stats
    });
});

// ======================
// KHỞI ĐỘNG
// ======================
async function start() {
    console.log('========================================');
    console.log('BACCARAT API SERVER + BCR-AML');
    console.log('Dự đoán từ phiên đầu — vừa đoán vừa học');
    console.log('========================================');

    aml.loadFromFile();

    console.log('[1] Đang đăng nhập...');
    const loginOk = await login();
    if (!loginOk) {
        console.error('[ERROR] Đăng nhập thất bại!');
        process.exit(1);
    }
    console.log('[OK] Đăng nhập thành công');

    console.log('[2] Vào lobby...');
    await goToLobby();
    console.log('[OK] Vào lobby thành công');

    console.log('[3] Lấy dữ liệu lần đầu...');
    await fetchBaccaratData();
    console.log(`[OK] Đã lấy ${baccaratData.length} bàn`);

    console.log('\n📊 DANH SÁCH BÀN:');
    baccaratData.forEach(item => {
        const rs = item.result.substring(0, 30) + (item.result.length > 30 ? '...' : '');
        console.log(`   ${item.table.padEnd(4)}: ${rs}`);
    });

    console.log('\n🤖 AML DỰ ĐOÁN BAN ĐẦU:');
    aml.getAllPredictions().forEach(p => {
        console.log(`   ${String(p['bàn']).padEnd(4)} | phiên=${String(p['phiên']).padEnd(4)} | cầu=${String(p['cầu']).padEnd(14)} | dự đoán=${p['dự đoán']} | tỉ lệ=${p['tỉ lệ']}`);
    });

    // Lưu AML state mỗi 60 giây
    setInterval(() => aml.saveToFile(), 60000);

    // Chạy auto update background
    autoUpdate();

    const PORT = process.env.PORT || 5000;
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`\n🚀 API SERVER ĐANG CHẠY TRÊN PORT ${PORT}`);
        console.log(`\n📡 Dữ liệu thô:`);
        console.log(`   http://localhost:${PORT}/api/baccarat`);
        console.log(`   http://localhost:${PORT}/api/baccarat/C01`);
        console.log(`   http://localhost:${PORT}/api/latest`);
        console.log(`\n🤖 Dự đoán AML:`);
        console.log(`   http://localhost:${PORT}/api/predict`);
        console.log(`   http://localhost:${PORT}/api/predict/C01`);
        console.log(`   http://localhost:${PORT}/api/predict/C01/detail`);
        console.log(`\n⏰ Auto update mỗi 2s | AML state lưu mỗi 60s`);
    });
}

start();
