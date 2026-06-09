// Static data exported from the AgentCIBench release artifacts.
window.AGENTS = [
  { model:"Claude-Opus-4.7",    family:"anthropic", group:"prop", util:81.2, leak:13.68, refusal:1.71, eng:13.91, ci:[7.7,20.5] },
  { model:"GPT-5.4",            family:"openai",    group:"prop", util:44.44,leak:18.8,  refusal:41.88,eng:32.35, ci:[12.0,26.5] },
  { model:"Claude-Sonnet-4.6",  family:"anthropic", group:"prop", util:52.14,leak:46.15, refusal:15.38,eng:54.55, ci:[37.6,54.7] },
  { model:"GPT-5.4-mini",       family:"openai",    group:"prop", util:52.14,leak:60.68, refusal:6.84, eng:65.14, ci:[52.1,69.2] },
  { model:"Grok-4.3",           family:"xai",       group:"prop", util:78.63,leak:91.45, refusal:0.0,  eng:91.45, ci:[86.3,95.7] },
  { model:"Gemini-3-Flash",     family:"google",    group:"prop", util:88.03,leak:93.16, refusal:0.85, eng:93.97, ci:[88.0,97.4] },
  { model:"Qwen-3.6-Max",       family:"qwen",      group:"prop", util:82.05,leak:97.44, refusal:0.0,  eng:97.44, ci:[94.0,100] },
  { model:"Gemini-3.1-Pro",     family:"google",    group:"prop", util:96.58,leak:98.29, refusal:0.0,  eng:98.29, ci:[95.7,100] },

  { model:"MiniMax-M2.7",       family:"minimax",   group:"open", util:68.38,leak:58.12, refusal:8.55, eng:63.55, ci:[48.7,66.7] },
  { model:"Gemma-4-26B",        family:"google",    group:"open", util:74.36,leak:67.52, refusal:2.56, eng:69.30, ci:[59.0,76.1] },
  { model:"Qwen-3.6-35B-A3B",   family:"qwen",      group:"open", util:82.91,leak:76.92, refusal:0.85, eng:77.59, ci:[69.2,84.6] },
  { model:"GPT-OSS-120B",       family:"openai",    group:"open", util:65.81,leak:65.81, refusal:16.24,eng:78.57, ci:[57.3,74.4] },
  { model:"Kimi-K2.6",          family:"moonshot",  group:"open", util:43.59,leak:62.39, refusal:22.22,eng:80.22, ci:[53.8,70.9] },
  { model:"DeepSeek-v4-Pro",    family:"deepseek",  group:"open", util:64.1, leak:82.91, refusal:2.56, eng:85.09, ci:[76.1,89.7] },
  { model:"GLM-5.1",            family:"zhipu",     group:"open", util:57.26,leak:85.47, refusal:4.27, eng:89.29, ci:[78.6,91.5] },
];

window.FAMILY_COLORS = {
  anthropic:"#cc785c", openai:"#10a37f", google:"#4285f4", xai:"#000000",
  qwen:"#a020f0", moonshot:"#1e6cff", deepseek:"#5b6cff", minimax:"#ff6b3d",
  zhipu:"#00b3a4"
};

// Defense macro averages from the paper artifacts.
window.DEFENSES = [
  { def:"none",            util:63.25, eng:51.7 },
  { def:"restrictive",     util:78.92, eng:19.0 },
  { def:"rubric-informed", util:79.20, eng:15.8 },
  { def:"recipient-typed", util:86.32, eng:16.2 },
];

// End-to-end transfer summary.
window.E2E = [
  { model:"Claude-Opus-4.7",   family:"anthropic", sg_eng:14.0, e2e_eng:42.9, leaks:6, engaged_n:14 },
  { model:"Claude-Sonnet-4.6", family:"anthropic", sg_eng:54.5, e2e_eng:80.0, leaks:8, engaged_n:10 },
];
