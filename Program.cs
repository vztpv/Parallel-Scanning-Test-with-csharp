using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading;
using System.Threading.Tasks;


// Token = uint32  (버퍼 내 시작 위치) 
using Token = System.UInt32;

namespace Clau
{
    // ─── 토큰 타입 ───────────────────────────────────────────────────────────────
    public enum TokenType
    {
        LeftBrace, RightBrace, LeftBracket, RightBracket,
        Assignment, Comma, Colon,
        BackSlash, Quoted,
        String, Number, True, False, Null, End
    }


    // ─── 상수 ────────────────────────────────────────────────────────────────────
    public static class LoadDataOption
    {
        public const char LeftBrace = '{';
        public const char RightBrace = '}';
        public const char LeftBracket = '[';
        public const char RightBracket = ']';
        public const char Assignment = ':';
        public const char Comma = ',';
    }

    // ─── Utility ─────────────────────────────────────────────────────────────────
    public static class Utility
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsWhitespace(char ch) =>
            ch is ' ' or '\t' or '\r' or '\n' or '\v' or '\f';

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Token Get(long position, long length, ReadOnlySpan<byte> _)
            => (Token)position;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TokenType GetType(char ch) => ch switch
        {
            '{' => TokenType.LeftBrace,
            '}' => TokenType.RightBrace,
            '[' => TokenType.LeftBracket,
            ']' => TokenType.RightBracket,
            ':' => TokenType.Assignment,
            ',' => TokenType.Comma,
            '\\' => TokenType.BackSlash,
            '"' => TokenType.Quoted,
            _ => TokenType.String,
        };

        public static void PrintToken(TextWriter out_, byte[] buffer, Token token, Token nextToken)
        {
            uint len = nextToken - token;
            out_.Write(Encoding.UTF8.GetString(buffer, (int)token, (int)len));
        }

        // BOM 감지 (UTF-8 BOM: EF BB BF)
        public static long SkipBom(byte[] data)
        {
            if (data.Length >= 3 &&
                data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF)
                return 3;
            return 0;
        }
    }

    // ─── SIMD 헬퍼 (AVX2) ────────────────────────────────────────────────────────
    public static unsafe class SimdHelper
    {
        public static uint GetDelimiterMaskAvx2(Vector256<byte> chunk)
        {
            var quote = Avx2.CompareEqual(chunk, Vector256.Create((byte)'"'));
            var slash = Avx2.CompareEqual(chunk, Vector256.Create((byte)'\\'));
            var lb = Avx2.CompareEqual(chunk, Vector256.Create((byte)'{'));
            var rb = Avx2.CompareEqual(chunk, Vector256.Create((byte)'}'));
            var lbk = Avx2.CompareEqual(chunk, Vector256.Create((byte)'['));
            var rbk = Avx2.CompareEqual(chunk, Vector256.Create((byte)']'));
            var colon = Avx2.CompareEqual(chunk, Vector256.Create((byte)':'));
            var comma = Avx2.CompareEqual(chunk, Vector256.Create((byte)','));

            var structural = Avx2.Or(
                Avx2.Or(Avx2.Or(lb, rb), Avx2.Or(lbk, rbk)),
                Avx2.Or(colon, comma));

            var sp = Avx2.CompareEqual(chunk, Vector256.Create((byte)' '));
            var nl = Avx2.CompareEqual(chunk, Vector256.Create((byte)'\n'));
            var cr = Avx2.CompareEqual(chunk, Vector256.Create((byte)'\r'));
            var tab = Avx2.CompareEqual(chunk, Vector256.Create((byte)'\t'));
            var ws = Avx2.Or(Avx2.Or(sp, nl), Avx2.Or(cr, tab));

            var all = Avx2.Or(Avx2.Or(quote, slash), Avx2.Or(structural, ws));

            return (uint)Avx2.MoveMask(all);
        }
    }

    // ─── InFileReserver ───────────────────────────────────────────────────────────
    public sealed class InFileReserver
    {
        private byte[]? _buffer;
        private Token[]? _tokenOrig;  // 단일 연속 버퍼 (C++ tokens_orig 대응)

        // ── Stage 1: SIMD 스캔 ───────────────────────────────────────────────────
        // tokenOffset: _tokenOrig 내 이 세그먼트의 시작 위치 (C++ 포인터 오프셋 대응)
        private static unsafe void ScanWithSimdStyle(
            byte[] text, long num, long length,
            Token[] tokenArr, long tokenOffset,
            out long tokenArrSize, out long quotedCount)
        {
            long tokenFirst = 0;
            long tokenCount = tokenOffset;  // ★ 오프셋부터 시작
            long qCount = 0;
            int backslashOn = -1;
            long i = 0;

            void FlushWord(long endIdx)
            {
                if (endIdx > tokenFirst)
                    tokenArr[tokenCount++] = Utility.Get(tokenFirst + num, endIdx - tokenFirst, default);
            }

            bool useAvx2 = Avx2.IsSupported;

            fixed (byte* ptr = text)
            {
                if (useAvx2)
                {
                    while (i + 32 <= length)
                    {
                        var chunk = Avx2.LoadVector256(ptr + i);
                        uint mask = SimdHelper.GetDelimiterMaskAvx2(chunk);

                        while (mask != 0)
                        {
                            uint bitIdx = (uint)System.Numerics.BitOperations.TrailingZeroCount(mask);
                            mask &= mask - 1;

                            if (backslashOn >= 0)
                            {
                                if (i + bitIdx == (uint)backslashOn) { backslashOn = -1; continue; }
                                backslashOn = -1;
                            }

                            long actualIdx = i + bitIdx;
                            char ch = (char)text[actualIdx];

                            FlushWord(actualIdx);

                            if (ch is ' ' or '\n' or '\r' or '\t')
                            {
                                tokenFirst = actualIdx + 1;
                            }
                            else if (ch == '\\')
                            {
                                tokenFirst = actualIdx;
                                backslashOn = (int)(actualIdx + 1);
                            }
                            else
                            {
                                if (ch == '"') qCount++;
                                tokenArr[tokenCount++] = Utility.Get(actualIdx + num, 1, default);
                                tokenFirst = actualIdx + 1;
                            }
                        }
                        i += 32;
                    }
                }

                while (i < length)
                {
                    char ch = (char)text[i];
                    switch (ch)
                    {
                        case ' ':
                        case '\t':
                        case '\r':
                        case '\v':
                        case '\f':
                        case '\n':
                            FlushWord(i);
                            tokenFirst = i + 1;
                            break;
                        case '"':
                            ++qCount;
                            FlushWord(i);
                            tokenArr[tokenCount++] = Utility.Get(i + num, 1, default);
                            tokenFirst = i + 1;
                            break;
                        case ',':
                            FlushWord(i);
                            tokenArr[tokenCount++] = Utility.Get(i + num, 1, default);
                            tokenFirst = i + 1;
                            break;
                        case '\\':
                            FlushWord(i);
                            if (i + 1 < length) { ++i; tokenFirst = i - 1; }
                            else { tokenFirst = i; }
                            break;
                        case '{':
                        case '[':
                        case '}':
                        case ']':
                        case ':':
                            FlushWord(i);
                            tokenArr[tokenCount++] = Utility.Get(i + num, 1, default);
                            tokenFirst = i + 1;
                            break;
                    }
                    ++i;
                }
            }

            FlushWord(length);
            tokenArrSize = tokenCount - tokenOffset;  // ★ 이 세그먼트에서 쓴 개수만
            quotedCount = qCount;
        }

        // ── Stage 2: 따옴표 쌍 병합 ─────────────────────────────────────────────
        // tokenOffset: _tokenOrig 내 이 세그먼트의 시작 위치
        private static void MergeQuotes(
            byte[] text, long num, long length,
            Token[] tokenArr, long tokenOffset,
            long tokenArrSize,
            out long outSize, out int lastState,
            long precedingQuoteCount)
        {
            int state = (int)(precedingQuoteCount % 2);
            long startToken = tokenOffset;
            long count = tokenOffset;  // ★ 오프셋부터 덮어씀

            long end = tokenOffset + tokenArrSize;
            for (long p = tokenOffset; p < end; p++)
            {
                char ch = (char)text[tokenArr[p] - num];
                var tt = Utility.GetType(ch);

                if (state == 0)
                {
                    if (tt == TokenType.Quoted)
                    {
                        state = 1;
                        startToken = p;
                    }
                    else if (tt != TokenType.BackSlash)
                    {
                        tokenArr[count++] = tokenArr[p];
                    }
                }
                else
                {
                    if (tt == TokenType.Quoted)
                    {
                        tokenArr[count++] = tokenArr[startToken];
                        state = 0;
                    }
                }
            }

            if (state == 1)
                tokenArr[count++] = tokenArr[startToken];

            outSize = count - tokenOffset;  // ★ 이 세그먼트에서 쓴 개수만
            lastState = state;
        }

        // ── 병렬 스캔 조합 ───────────────────────────────────────────────────────
        private bool ScanningNew(
            byte[] text, long length, int thrNum,
            out long[]? segOffsets, out long[]? tokenSize, out long totalTokenCount)
        {
            segOffsets = null;
            tokenSize = null;
            totalTokenCount = 0;

            // ① 스레드 분할 경계 계산
            var starts = new List<long> { 0 };
            for (int i = 1; i < thrNum; i++)
            {
                long pos = length / thrNum * i;
                while (pos < length)
                {
                    char c = (char)text[pos];
                    if (Utility.IsWhitespace(c) ||
                        c is '{' or '}' or '[' or ']' or ',' or ':')
                    { starts.Add(pos); break; }
                    pos++;
                    if (pos == length) return false;
                }
            }

            var startSet = new SortedSet<long>(starts);
            starts = new List<long>(startSet);
            thrNum = starts.Count;

            var startsArr = starts.ToArray();
            var lastsArr = new long[thrNum];
            for (int i = 0; i < thrNum - 1; i++) lastsArr[i] = startsArr[i + 1];
            lastsArr[thrNum - 1] = length;

            // ② 단일 버퍼 확보 + 오프셋 계산
            //    C++ : tokens[i] = tokens[i-1] + (last[i-1] - start[i-1])
            //    C#  : offsets[i] = offsets[i-1] + (last[i-1] - start[i-1])
            var offsets = new long[thrNum];
            long needed = 0;
            for (int i = 0; i < thrNum; i++)
            {
                offsets[i] = needed;
                needed += (lastsArr[i] - startsArr[i]) + 1;  // +1: 보초 토큰 공간
            }

            if (_tokenOrig == null || _tokenOrig.Length < needed)
                _tokenOrig = new Token[needed];

            var tokenSizes = new long[thrNum];
            var quoteCounts = new long[thrNum];
            var lastStates = new int[thrNum];

            // 람다 캡처용 로컬 참조
            var tokenOrig = _tokenOrig;

            // ③ Stage-1 병렬 실행
            var t1 = DateTime.UtcNow;
            Parallel.For(0, thrNum, i =>
            {
                long segLen = lastsArr[i] - startsArr[i];
                ScanWithSimdStyle(
                    text, startsArr[i], segLen,
                    tokenOrig, offsets[i],       // ★ 단일 배열 + 오프셋
                    out tokenSizes[i], out quoteCounts[i]);
            });
            var t2 = DateTime.UtcNow;

            // ④ 누적 따옴표 수
            for (int i = 1; i < thrNum; i++)
                quoteCounts[i] += quoteCounts[i - 1];

            // ⑤ Stage-2 병렬 실행
            Parallel.For(0, thrNum, i =>
            {
                long preceding = i == 0 ? 0 : quoteCounts[i - 1];
                long segLen = lastsArr[i] - startsArr[i];
                MergeQuotes(
                    text, startsArr[i], segLen,
                    tokenOrig, offsets[i],       // ★ 단일 배열 + 오프셋
                    tokenSizes[i],
                    out tokenSizes[i], out lastStates[i], preceding);
            });
            var t3 = DateTime.UtcNow;

            // ⑥ 세그먼트 경계 보초 토큰 설정
            for (int t = 0; t < thrNum; t++)
            {
                long sentinelPos = offsets[t] + tokenSizes[t];
                if (t < thrNum - 1 && tokenSizes[t] > 0)
                    tokenOrig[sentinelPos] = tokenOrig[offsets[t + 1]];  // 다음 세그 첫 토큰
                else if (tokenSizes[t] > 0)
                    tokenOrig[sentinelPos] = (Token)length;
            }

            Console.WriteLine($"토큰 후보 배열 구성(parallel)\t{(t2 - t1).TotalMilliseconds:F0}ms");
            Console.WriteLine($"토큰 배열 구성(parallel)\t{(t3 - t2).TotalMilliseconds:F0}ms");

            long total = 0;
            foreach (var s in tokenSizes) total += s;

            segOffsets = offsets;
            tokenSize = tokenSizes;
            totalTokenCount = total;
            return true;
        }

        // ── 파일 스캔 진입점 ─────────────────────────────────────────────────────
        private bool Scan(string fileName, int thrNum,
            out long[]? segOffsets, out long[]? tokenSize, out long tokenArrayLen)
        {
            segOffsets = null;
            tokenArrayLen = 0;
            tokenSize = null;

            int t0 = Environment.TickCount;
            try { _buffer = File.ReadAllBytes(fileName); }
            catch { Console.WriteLine("file read fail"); return false; }

            long bomOffset = Utility.SkipBom(_buffer);
            long fileLen = _buffer.Length - bomOffset;
            byte[] text = bomOffset == 0 ? _buffer : _buffer[(int)bomOffset..];

            Console.WriteLine($"load file\t{Environment.TickCount - t0}ms\tfile size {fileLen}");

            return ScanningNew(text, fileLen, thrNum,
                out segOffsets, out tokenSize, out tokenArrayLen);
        }

        // ── 공개 호출 ─────────────────────────────────────────────────────────────
        public bool Invoke(string fileName, int thrNum,
            out long[]? segOffsets, out long[]? tokenSize, out long tokenArrayLen)
            => Scan(fileName, thrNum, out segOffsets, out tokenSize, out tokenArrayLen);

        // 파서에서 토큰 값을 읽을 때 사용
        public Token[]? TokenOrig => _tokenOrig;

        public void Dispose() { }
    }

    // ─── LoadData ─────────────────────────────────────────────────────────────────
    public sealed class LoadData
    {
        private readonly InFileReserver _ifReserver = new();

        public bool LoadDataFromFile(
            string fileName,
            int lexThrNum = 1,
            int parseThrNum = 1,
            bool useSimd = false)
        {
            if (lexThrNum <= 0) lexThrNum = Environment.ProcessorCount;
            if (parseThrNum <= 0) parseThrNum = Environment.ProcessorCount;

            int a = Environment.TickCount;
            try
            {
                _ifReserver.Invoke(fileName, lexThrNum,
                    out var segOffsets, out var tokenSize, out var totalTokens);

                int b = Environment.TickCount;
                Console.WriteLine($"total tokens: {totalTokens}\t{b - a}ms");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                return false;
            }
            return true;
        }
    }

    // ─── Program ─────────────────────────────────────────────────────────────────
    public static class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: JsonParser <file>");
                return;
            }

            var test = new LoadData();
            for (int i = 0; i < 10; i++)
            {
                int a = Environment.TickCount;
                test.LoadDataFromFile(args[0], 0, 0, true);
                int b = Environment.TickCount;
                Console.WriteLine($"test end {b - a}ms");
            }
        }
    }
}