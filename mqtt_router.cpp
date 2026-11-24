/*
 * mqtt_router.cpp - Ultra lightweight MQTT topic router / rules engine.
 *
 * Operation:
 *   Subscribes to incoming MQTT topics defined in an INI file, evaluates
 *   payloads against per-rule conditions, and publishes outgoing messages.
 *   Supports MQTT wildcards (+, #), per-rule cooldowns, edge-triggering
 *   (fire only on false->true transition), payload range checks, and live 
 *   config reload via SIGHUP.
 *
 * Build & Run:
 *   g++ -std=c++17 -O2 -Wall -Wextra mqtt_router.cpp -lmosquitto -o mqtt_router
 *   ./mqtt_router <rules.ini>
 *   kill -HUP $(pidof mqtt_router)   # reload rules without restarting
 *
 * INI Format:
 *   Lines starting with '#' or ';' are comments.
 *
 *   [broker]                         # Optional (defaults: localhost:1883)
 *   host = <hostname>
 *   port = <port>
 *
 *   [<rule_id>]                      # Any other section defines a rule
 *   in_topic     = <mqtt_topic>      # Required; wildcards (+, #) supported
 *   op           = <operator>        # Optional (default: any)
 *                                    #   any           - always matches (even empty payload)
 *                                    #   equals        - exact match against 'value'
 *                                    #   contains      - substring match against 'value'
 *                                    #   outside_range - numeric payload < min or > max
 *                                    #   in_range      - numeric payload >= min and <= max
 *                                    #   gt / lt       - numeric payload > max / < min
 *   value        = <string>          # Used by equals, contains
 *   min          = <number>          # Used by outside_range, in_range, lt
 *   max          = <number>          # Used by outside_range, in_range, gt
 *   cooldown     = <seconds>         # Optional; min seconds between fires (default: 0)
 *   edge_trigger = true|false        # Optional; fire only on state change (default: false)
 *   out          = <topic> | <payload>  # Repeatable; omit '| <payload>' for empty payload
 */

#include <mosquitto.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

volatile sig_atomic_t g_running = 1;
volatile sig_atomic_t g_reload  = 0;

void handle_signal(int sig) {
    if (sig == SIGHUP) {
        g_reload = 1;
    } else {
        g_running = 0;
    }
}

std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

enum class Op { Any, Equals, Contains, OutsideRange, InRange, Gt, Lt };

struct Output {
    std::string topic;
    std::string payload;
};

struct Rule {
    std::string id;
    std::string in_topic;
    Op op = Op::Any;
    std::string value;
    double min_val = 0.0;
    double max_val = 0.0;
    int cooldown_sec = 0;
    bool edge_trigger = false;
    std::vector<Output> outputs;

    // Runtime state
    bool last_matched = false;
    bool has_fired = false;
    std::chrono::steady_clock::time_point last_fired;
};

struct Config {
    std::string host = "localhost";
    int port = 1883;
    std::vector<Rule> rules;
};

bool parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && trim(end).empty();
}

bool load_config(const std::string& path, Config& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[error] Could not open config: " << path << "\n";
        return false;
    }

    Config next;
    std::string line;
    std::string section;
    Rule* current = nullptr;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;

        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            if (section == "broker") {
                current = nullptr;
            } else {
                next.rules.push_back(Rule{});
                current = &next.rules.back();
                current->id = section;
            }
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));

        if (section == "broker") {
            if (key == "host") next.host = val;
            else if (key == "port") next.port = std::atoi(val.c_str());
        } else if (current) {
            if (key == "in_topic") current->in_topic = val;
            else if (key == "value") current->value = val;
            else if (key == "min") parse_double(val, current->min_val);
            else if (key == "max") parse_double(val, current->max_val);
            else if (key == "cooldown") current->cooldown_sec = std::atoi(val.c_str());
            else if (key == "edge_trigger") current->edge_trigger = (val == "true" || val == "1");
            else if (key == "op") {
                if (val == "equals") current->op = Op::Equals;
                else if (val == "contains") current->op = Op::Contains;
                else if (val == "outside_range") current->op = Op::OutsideRange;
                else if (val == "in_range") current->op = Op::InRange;
                else if (val == "gt") current->op = Op::Gt;
                else if (val == "lt") current->op = Op::Lt;
                else current->op = Op::Any;
            } else if (key == "out") {
                const auto pipe = val.find('|');
                if (pipe == std::string::npos) {
                    current->outputs.push_back({val, ""});
                } else {
                    current->outputs.push_back({
                        trim(val.substr(0, pipe)),
                        trim(val.substr(pipe + 1))
                    });
                }
            }
        }
    }

    cfg = std::move(next);
    return true;
}

bool evaluate_rule(const Rule& r, const std::string& payload) {
    switch (r.op) {
        case Op::Any:      return true;
        case Op::Equals:   return payload == r.value;
        case Op::Contains: return payload.find(r.value) != std::string::npos;
        case Op::OutsideRange:
        case Op::InRange:
        case Op::Gt:
        case Op::Lt: {
            double num = 0.0;
            if (!parse_double(payload, num)) return false;
            if (r.op == Op::OutsideRange) return num < r.min_val || num > r.max_val;
            if (r.op == Op::InRange)      return num >= r.min_val && num <= r.max_val;
            if (r.op == Op::Gt)           return num > r.max_val;
            if (r.op == Op::Lt)           return num < r.min_val;
            return false;
        }
    }
    return false;
}

struct RouterState {
    Config config;
    std::set<std::string> active_subs;
    bool connected = false;
};

void sync_subscriptions(struct mosquitto* mosq, RouterState& state) {
    if (!state.connected) return;

    std::set<std::string> desired;
    for (const auto& r : state.config.rules) {
        if (!r.in_topic.empty()) desired.insert(r.in_topic);
    }

    for (const auto& old_sub : state.active_subs) {
        if (desired.find(old_sub) == desired.end()) {
            mosquitto_unsubscribe(mosq, nullptr, old_sub.c_str());
        }
    }
    for (const auto& new_sub : desired) {
        if (state.active_subs.find(new_sub) == state.active_subs.end()) {
            mosquitto_subscribe(mosq, nullptr, new_sub.c_str(), 0);
        }
    }
    state.active_subs = std::move(desired);
}

void on_connect(struct mosquitto* mosq, void* userdata, int rc) {
    auto* state = static_cast<RouterState*>(userdata);
    if (rc == 0) {
        std::cout << "[info] Connected to MQTT broker.\n";
        state->connected = true;
        state->active_subs.clear();  // Re-subscribe everything on reconnect
        sync_subscriptions(mosq, *state);
    } else {
        std::cerr << "[error] Broker connection failed (rc=" << rc << ")\n";
    }
}

void on_disconnect(struct mosquitto*, void* userdata, int) {
    auto* state = static_cast<RouterState*>(userdata);
    state->connected = false;
    std::cerr << "[warn] Disconnected from broker.\n";
}

void on_message(struct mosquitto* mosq, void* userdata, const struct mosquitto_message* msg) {
    auto* state = static_cast<RouterState*>(userdata);
    const std::string topic = msg->topic ? msg->topic : "";
    const std::string payload = (msg->payload && msg->payloadlen > 0)
        ? std::string(static_cast<const char*>(msg->payload), msg->payloadlen)
        : "";

    const auto now = std::chrono::steady_clock::now();

    for (auto& rule : state->config.rules) {
        bool topic_match = false;
        mosquitto_topic_matches_sub(rule.in_topic.c_str(), topic.c_str(), &topic_match);
        if (!topic_match) continue;

        const bool matched = evaluate_rule(rule, payload);
        const bool should_fire = rule.edge_trigger ? (matched && !rule.last_matched) : matched;
        rule.last_matched = matched;

        if (!should_fire) continue;

        if (rule.has_fired && rule.cooldown_sec > 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - rule.last_fired).count();
            if (elapsed < rule.cooldown_sec) continue;
        }

        rule.last_fired = now;
        rule.has_fired = true;

        std::cout << "[fire] Rule '" << rule.id << "' triggered by " << topic << "\n";
        for (const auto& out : rule.outputs) {
            mosquitto_publish(
                mosq, nullptr, out.topic.c_str(),
                static_cast<int>(out.payload.size()),
                out.payload.empty() ? nullptr : out.payload.data(),
                0, false
            );
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rules.ini>\n";
        return 1;
    }
    const std::string config_path = argv[1];

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGHUP, handle_signal);

    RouterState state;
    if (!load_config(config_path, state.config)) {
        return 1;
    }

    mosquitto_lib_init();
    struct mosquitto* mosq = mosquitto_new("mqtt_router", true, &state);
    if (!mosq) {
        std::cerr << "[error] Failed to allocate mosquitto client.\n";
        return 1;
    }

    mosquitto_connect_callback_set(mosq, on_connect);
    mosquitto_disconnect_callback_set(mosq, on_disconnect);
    mosquitto_message_callback_set(mosq, on_message);

    if (mosquitto_connect(mosq, state.config.host.c_str(), state.config.port, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr << "[warn] Initial connect failed, will retry...\n";
    }

    while (g_running) {
        const int rc = mosquitto_loop(mosq, 100, 1);
        if (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST) {
            mosquitto_reconnect(mosq);
        }

        if (g_reload) {
            g_reload = 0;
            std::cout << "[info] SIGHUP received, reloading " << config_path << "\n";
            if (load_config(config_path, state.config)) {
                sync_subscriptions(mosq, state);
                std::cout << "[info] Loaded " << state.config.rules.size() << " rules.\n";
            }
        }
    }

    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
