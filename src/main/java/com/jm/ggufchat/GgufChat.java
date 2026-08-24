package com.jm.ggufchat;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.args.LogFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Locale;

/**
 * Interactive chat over a GGUF model using the de.kherud:llama Java bindings
 * (which wrap llama.cpp / GGML).
 *
 * <p>The Llama-3 chat template is applied on the Java side by building the
 * exact header / eot markup the tokenizer would produce, so the model sees
 * a correctly-formatted prompt regardless of how the upstream
 * {@code setMessages + setUseChatTemplate} path behaves in this build.
 *
 * <p>Usage:
 * <pre>
 *   mvn -q exec:java                              # uses default model in cwd
 *   mvn -q exec:java -Dexec.args="--model path.gguf --ctx 4096"
 *   mvn -q exec:java -Dexec.args="--prompt 'Tell me a joke' --exit-after"
 * </pre>
 */
public final class GgufChat {

    private static final String DEFAULT_MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    private static final String DEFAULT_SYSTEM =
            "You are a helpful, accurate, and concise assistant running locally on the user's machine.";

    private final Cli cli;
    private final Deque<Message> history = new ArrayDeque<>();
    private String systemPrompt = DEFAULT_SYSTEM;

    public static void main(String[] args) throws Exception {
        Cli cli = Cli.parse(args);
        try {
            new GgufChat(cli).run();
        } finally {
            // The native lib spawns non-daemon worker threads that keep the
            // JVM alive after the model is closed. Force-exit cleanly.
            System.exit(0);
        }
    }

    private GgufChat(Cli cli) {
        this.cli = cli;
    }

    private void run() throws Exception {
        Path modelPath = resolveModel(cli.model);
        if (!Files.isRegularFile(modelPath)) {
            throw new IOException("Model file not found: " + modelPath.toAbsolutePath());
        }

        banner(modelPath);

        if (cli.quiet) {
            LlamaModel.setLogger(null, (level, msg) -> {});
        } else {
            LlamaModel.setLogger(LogFormat.JSON, null);
        }

        ModelParameters mp = new ModelParameters()
                .setModel(modelPath.toAbsolutePath().toString())
                .setCtxSize(cli.ctxSize)
                .setThreads(cli.threads)
                .setGpuLayers(cli.gpuLayers);

        long t0 = System.currentTimeMillis();
        try (LlamaModel model = new LlamaModel(mp)) {
            long loadMs = System.currentTimeMillis() - t0;
            System.out.printf("Model loaded in %.2f s. ctx=%d, threads=%d, gpu-layers=%d%n",
                    loadMs / 1000.0, cli.ctxSize, cli.threads, cli.gpuLayers);
            System.out.println();

            if (cli.system != null) {
                this.systemPrompt = cli.system;
            }
            if (cli.system != null) {
                System.out.println("[system] " + systemPrompt);
            }

            if (cli.oneShotPrompt != null) {
                runOneShot(model, cli.oneShotPrompt);
                return;
            }
            runRepl(model);
        }
    }

    private void runOneShot(LlamaModel model, String prompt) {
        history.addLast(Message.user(prompt));
        String reply = generate(model);
        System.out.println();
        System.out.println("[assistant] " + reply);
    }

    private void runRepl(LlamaModel model) throws IOException {
        System.out.println("Type /help for commands. /exit or Ctrl+D to quit.");
        System.out.println();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        while (true) {
            System.out.print("you> ");
            String line;
            try {
                line = in.readLine();
            } catch (IOException e) {
                System.out.println();
                return;
            }
            if (line == null) {
                System.out.println();
                return;
            }
            String input = line.trim();
            if (input.isEmpty()) {
                continue;
            }
            if (input.startsWith("/")) {
                if (handleCommand(model, input)) {
                    return;
                }
                continue;
            }
            history.addLast(Message.user(input));
            String reply = generate(model);
            history.addLast(Message.assistant(reply));
        }
    }

    private boolean handleCommand(LlamaModel model, String cmd) {
        String[] parts = cmd.split("\\s+", 2);
        String name = parts[0].toLowerCase(Locale.ROOT);
        String rest = parts.length > 1 ? parts[1] : "";
        switch (name) {
            case "/exit", "/quit" -> {
                return true;
            }
            case "/help" -> {
                System.out.println("""
                        Commands:
                          /help                 show this help
                          /reset                clear conversation history
                          /system <text>        set or replace the system prompt
                          /temp <float>         set sampling temperature (0 = greedy)
                          /ctx <int>            set context size (applies on next load)
                          /info                 print current settings
                          /save <file>          append last assistant reply to a file
                          /exit | /quit         leave the chat
                        """);
            }
            case "/reset" -> {
                history.clear();
                System.out.println("[history cleared]");
            }
            case "/system" -> {
                systemPrompt = rest;
                System.out.println("[system updated] " + systemPrompt);
            }
            case "/temp" -> {
                try {
                    cli.temperature = Float.parseFloat(rest);
                    System.out.printf("[temperature=%.3f]%n", cli.temperature);
                } catch (NumberFormatException ex) {
                    System.out.println("usage: /temp <float>");
                }
            }
            case "/ctx" -> {
                System.out.println("Context size is fixed at model load. Restart with --ctx " + rest);
            }
            case "/info" -> {
                System.out.printf("model=%s%nctx=%d%nthreads=%d%ngpu-layers=%d%ntemperature=%.3f%nmax-tokens=%d%nmessages=%d%nsystem=%s%n",
                        cli.model, cli.ctxSize, cli.threads, cli.gpuLayers, cli.temperature,
                        cli.maxTokens, history.size(),
                        systemPrompt == null ? "(none)" : systemPrompt);
            }
            case "/save" -> {
                if (rest.isBlank()) {
                    System.out.println("usage: /save <file>");
                    break;
                }
                Message last = lastAssistant();
                if (last == null) {
                    System.out.println("no assistant reply to save yet");
                    break;
                }
                try {
                    Files.writeString(Paths.get(rest), last.content() + System.lineSeparator(),
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.APPEND);
                    System.out.println("[saved -> " + rest + "]");
                } catch (IOException e) {
                    System.out.println("save failed: " + e.getMessage());
                }
            }
            default -> System.out.println("unknown command: " + name + " (try /help)");
        }
        return false;
    }

    private Message lastAssistant() {
        Message last = null;
        for (Message m : history) {
            if (m.role().equals("assistant")) {
                last = m;
            }
        }
        return last;
    }

    private String generate(LlamaModel model) {
        String prompt = buildLlamaChatPrompt(systemPrompt, history);

        InferenceParameters ip = new InferenceParameters(prompt)
                .setTemperature(cli.temperature)
                .setTopP(cli.topP)
                .setTopK(cli.topK)
                .setRepeatPenalty(cli.repeatPenalty)
                .setNPredict(cli.maxTokens)
                .setStopStrings("<|start_header_id|>", "<|eot_id|>")
                .setSeed(cli.seed);

        System.out.print("ai> ");
        StringBuilder sb = new StringBuilder();
        long t0 = System.nanoTime();
        int tokenCount = 0;
        for (LlamaOutput out : model.generate(ip)) {
            System.out.print(out.text);
            System.out.flush();
            sb.append(out.text);
            tokenCount++;
        }
        double secs = (System.nanoTime() - t0) / 1_000_000_000.0;
        double tps = secs > 0 ? tokenCount / secs : 0.0;
        System.out.println();
        System.out.printf("   [%d tokens, %.2f s, %.2f tok/s]%n", tokenCount, secs, tps);
        return sb.toString();
    }

    /**
     * Build a Llama-3 chat prompt. This format is what the model's built-in
     * chat template (tokenizer.chat_template) produces, so it matches what
     * the native template engine would emit. The tokenizer adds the leading
     * BOS (&lt;|begin_of_text|&gt;) automatically, so we don't prepend it.
     */
    private static String buildLlamaChatPrompt(String system, Deque<Message> turns) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
              .append(system.trim())
              .append("<|eot_id|>");
        }
        for (Message m : turns) {
            sb.append("<|start_header_id|>").append(m.role())
              .append("<|end_header_id|>\n\n")
              .append(m.content())
              .append("<|eot_id|>");
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private static Path resolveModel(String requested) {
        if (requested != null) {
            return Paths.get(requested);
        }
        Path cwd = Paths.get("").toAbsolutePath();
        Path candidate = cwd.resolve(DEFAULT_MODEL_NAME);
        if (Files.isRegularFile(candidate)) {
            return candidate;
        }
        return candidate;
    }

    private static void banner(Path modelPath) {
        long sizeBytes = -1;
        try {
            sizeBytes = Files.size(modelPath);
        } catch (IOException ignored) {
        }
        String sizeStr = sizeBytes > 0 ? "%.2f GB".formatted(sizeBytes / 1_000_000_000.0) : "?";
        System.out.println("=================================================");
        System.out.println(" gguf-chat  (Java + de.kherud:llama / GGML)");
        System.out.println(" model : " + modelPath.getFileName() + "  (" + sizeStr + ")");
        System.out.println(" path  : " + modelPath.toAbsolutePath());
        System.out.println(" java  : " + System.getProperty("java.version")
                + "  os=" + System.getProperty("os.name"));
        System.out.println("=================================================");
    }

    // ---- value types ----

    private record Message(String role, String content) {
        static Message user(String s) {
            return new Message("user", s);
        }

        static Message assistant(String s) {
            return new Message("assistant", s);
        }
    }

    // ---- CLI ----

    private static final class Cli {
        String model;
        int ctxSize = 4096;
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        int gpuLayers = 0;
        float temperature = 0.7f;
        float topP = 0.9f;
        int topK = 40;
        float repeatPenalty = 1.1f;
        int maxTokens = 1024;
        int seed = -1;
        String system;
        String oneShotPrompt;
        boolean quiet = true;

        static Cli parse(String[] args) {
            Cli c = new Cli();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                String next = (i + 1 < args.length) ? args[i + 1] : null;
                switch (a) {
                    case "--model" -> { c.model = require(next, a); i++; }
                    case "--ctx" -> { c.ctxSize = Integer.parseInt(require(next, a)); i++; }
                    case "--threads" -> { c.threads = Integer.parseInt(require(next, a)); i++; }
                    case "--gpu-layers" -> { c.gpuLayers = Integer.parseInt(require(next, a)); i++; }
                    case "--temperature", "--temp" -> { c.temperature = Float.parseFloat(require(next, a)); i++; }
                    case "--top-p" -> { c.topP = Float.parseFloat(require(next, a)); i++; }
                    case "--top-k" -> { c.topK = Integer.parseInt(require(next, a)); i++; }
                    case "--repeat-penalty" -> { c.repeatPenalty = Float.parseFloat(require(next, a)); i++; }
                    case "--max-tokens" -> { c.maxTokens = Integer.parseInt(require(next, a)); i++; }
                    case "--seed" -> { c.seed = Integer.parseInt(require(next, a)); i++; }
                    case "--system" -> { c.system = require(next, a); i++; }
                    case "--prompt" -> { c.oneShotPrompt = require(next, a); i++; }
                    case "--verbose" -> { c.quiet = false; }
                    case "-h", "--help" -> {
                        printHelp();
                        System.exit(0);
                    }
                    default -> {
                        if (a.startsWith("--")) {
                            System.err.println("unknown option: " + a);
                            printHelp();
                            System.exit(2);
                        }
                    }
                }
            }
            return c;
        }

        private static String require(String value, String name) {
            if (value == null || value.startsWith("--")) {
                throw new IllegalArgumentException("missing value for " + name);
            }
            return value;
        }

        private static void printHelp() {
            System.out.println("""
                    gguf-chat - Java chat over a GGUF model (de.kherud:llama / GGML)

                    Options:
                      --model <path>        GGUF model file (default: ./Llama-3.2-3B-Instruct-Q4_K_M.gguf)
                      --ctx <int>            context size in tokens (default: 4096)
                      --threads <int>        CPU threads (default: cores/2)
                      --gpu-layers <int>     layers to offload (default: 0 = CPU)
                      --temperature <float>  sampling temperature, 0 = greedy (default: 0.7)
                      --top-p <float>        nucleus sampling p (default: 0.9)
                      --top-k <int>          top-k sampling (default: 40)
                      --repeat-penalty <f>   repetition penalty (default: 1.1)
                      --max-tokens <int>     max new tokens per turn (default: 1024)
                      --seed <int>           RNG seed (default: random)
                      --system <text>        initial system prompt
                      --prompt <text>        one-shot prompt (prints reply and exits)
                      --verbose              show GGML/llama.cpp log output
                      --help                 this help

                    In the REPL: /help /reset /system <text> /temp <f> /info /save <file> /exit
                    """);
        }
    }
}
