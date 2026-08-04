#include "b70_provider.hpp"

#include <array>
#include <charconv>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <poll.h>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <unistd.h>

namespace {

using shooting_brake::phase1::B70Provider;
using shooting_brake::phase1::Capability;
using shooting_brake::phase1::Health;
using shooting_brake::phase1::ProviderConfig;
using shooting_brake::phase1::ProviderStatus;

constexpr std::string_view kDefaultBankPath = "phase1/expert_bank.bin";
constexpr int kSignalPollIntervalMs = 100;

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void request_stop(int) noexcept { stop_requested = 1; }

struct Options {
  std::string bank_path{kDefaultBankPath};
  std::size_t max_batch = 128;
  std::uint64_t generation = 1;
};

const char* status_name(const ProviderStatus status) noexcept {
  switch (status) {
    case ProviderStatus::ok:
      return "ok";
    case ProviderStatus::busy:
      return "busy";
    case ProviderStatus::not_loaded:
      return "not_loaded";
    case ProviderStatus::invalid_argument:
      return "invalid_argument";
    case ProviderStatus::generation_mismatch:
      return "generation_mismatch";
    case ProviderStatus::sequence_mismatch:
      return "sequence_mismatch";
    case ProviderStatus::device_error:
      return "device_error";
    case ProviderStatus::shutdown:
      return "shutdown";
  }
  return "unknown";
}

void write_json_string(std::ostream& output, const std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";

  output.put('"');
  for (std::size_t i = 0; i < value.size();) {
    const auto byte = static_cast<unsigned char>(value[i]);
    switch (byte) {
      case '"':
        output << "\\\"";
        ++i;
        continue;
      case '\\':
        output << "\\\\";
        ++i;
        continue;
      case '\b':
        output << "\\b";
        ++i;
        continue;
      case '\f':
        output << "\\f";
        ++i;
        continue;
      case '\n':
        output << "\\n";
        ++i;
        continue;
      case '\r':
        output << "\\r";
        ++i;
        continue;
      case '\t':
        output << "\\t";
        ++i;
        continue;
      default:
        break;
    }

    if (byte < 0x20) {
      output << "\\u00" << hex[byte >> 4] << hex[byte & 0x0f];
      ++i;
      continue;
    }
    if (byte < 0x80) {
      output.put(static_cast<char>(byte));
      ++i;
      continue;
    }

    std::size_t length = 0;
    std::uint32_t code_point = 0;
    std::uint32_t minimum = 0;
    if (byte >= 0xc2 && byte <= 0xdf) {
      length = 2;
      code_point = byte & 0x1f;
      minimum = 0x80;
    } else if (byte >= 0xe0 && byte <= 0xef) {
      length = 3;
      code_point = byte & 0x0f;
      minimum = 0x800;
    } else if (byte >= 0xf0 && byte <= 0xf4) {
      length = 4;
      code_point = byte & 0x07;
      minimum = 0x10000;
    }

    bool valid = length != 0 && i + length <= value.size();
    for (std::size_t offset = 1; valid && offset < length; ++offset) {
      const auto continuation = static_cast<unsigned char>(value[i + offset]);
      if ((continuation & 0xc0) != 0x80) {
        valid = false;
      } else {
        code_point = (code_point << 6) | (continuation & 0x3f);
      }
    }
    valid = valid && code_point >= minimum && code_point <= 0x10ffff &&
            !(code_point >= 0xd800 && code_point <= 0xdfff);

    if (valid) {
      output.write(value.data() + i, static_cast<std::streamsize>(length));
      i += length;
    } else {
      output << "\\ufffd";
      ++i;
    }
  }
  output.put('"');
}

void write_error(std::ostream& output, const std::string_view error,
                 const std::string_view detail = {},
                 const std::string_view status = {}) {
  output << "{\"error\":";
  write_json_string(output, error);
  if (!status.empty()) {
    output << ",\"status\":";
    write_json_string(output, status);
  }
  if (!detail.empty()) {
    output << ",\"detail\":";
    write_json_string(output, detail);
  }
  output << "}\n";
  output.flush();
}

template <typename Values>
void write_number_array(std::ostream& output, const Values& values) {
  output.put('[');
  bool first = true;
  for (const auto value : values) {
    if (!first) {
      output.put(',');
    }
    output << value;
    first = false;
  }
  output.put(']');
}

template <typename Values>
void write_string_array(std::ostream& output, const Values& values) {
  output.put('[');
  bool first = true;
  for (const auto& value : values) {
    if (!first) {
      output.put(',');
    }
    write_json_string(output, value);
    first = false;
  }
  output.put(']');
}

void write_capability(std::ostream& output, const Capability& capability) {
  output << "{\"protocol_version\":" << capability.protocol_version
         << ",\"device_name\":";
  write_json_string(output, capability.device_name);
  output << ",\"device_memory_total_bytes\":"
         << capability.device_memory_total_bytes
         << ",\"device_memory_available_bytes\":"
         << capability.device_memory_available_bytes << ",\"backend\":";
  write_json_string(output, capability.backend);
  output << ",\"supported_hidden_sizes\":";
  write_number_array(output, capability.supported_hidden_sizes);
  output << ",\"supported_intermediate_sizes\":";
  write_number_array(output, capability.supported_intermediate_sizes);
  output << ",\"supported_topk\":";
  write_number_array(output, capability.supported_topk);
  output << ",\"num_resident_experts\":"
         << capability.num_resident_experts
         << ",\"max_batch_remote\":" << capability.max_batch_remote
         << ",\"kernel_families\":";
  write_string_array(output, capability.kernel_families);
  output << ",\"health_heartbeat_interval_ms\":"
         << capability.health_heartbeat_interval_ms
         << ",\"num_layers\":" << capability.num_layers
         << ",\"experts_per_layer\":" << capability.experts_per_layer
         << "}\n";
  output.flush();
}

void write_health(std::ostream& output, const Health& health) {
  output << "{\"loaded\":" << (health.loaded ? "true" : "false")
         << ",\"pending\":" << (health.pending ? "true" : "false")
         << ",\"stopped\":" << (health.stopped ? "true" : "false")
         << ",\"generation\":" << health.generation
         << ",\"dispatches\":" << health.dispatches
         << ",\"allocations\":" << health.allocations
         << ",\"last_error\":";
  write_json_string(output, health.last_error);
  output << "}\n";
  output.flush();
}

template <typename Integer>
bool parse_positive_integer(const std::string_view text, Integer* value) {
  if (text.empty()) {
    return false;
  }

  Integer parsed = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed, 10);
  if (result.ec != std::errc{} || result.ptr != end || parsed == 0) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parse_options(const int argc, char* argv[], Options* options,
                   std::string* error) {
  bool saw_bank = false;
  bool saw_max_batch = false;
  bool saw_generation = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view argument{argv[i]};
    if (argument == "--bank") {
      if (saw_bank) {
        *error = "--bank may only be specified once";
        return false;
      }
      if (++i == argc || argv[i][0] == '\0') {
        *error = "--bank requires a non-empty path";
        return false;
      }
      options->bank_path = argv[i];
      saw_bank = true;
    } else if (argument == "--max-batch") {
      if (saw_max_batch) {
        *error = "--max-batch may only be specified once";
        return false;
      }
      if (++i == argc) {
        *error = "--max-batch requires a positive integer";
        return false;
      }
      std::uint64_t parsed = 0;
      if (!parse_positive_integer(std::string_view{argv[i]}, &parsed) ||
          parsed > std::numeric_limits<std::size_t>::max()) {
        *error = "--max-batch requires a positive integer in range";
        return false;
      }
      options->max_batch = static_cast<std::size_t>(parsed);
      saw_max_batch = true;
    } else if (argument == "--generation") {
      if (saw_generation) {
        *error = "--generation may only be specified once";
        return false;
      }
      if (++i == argc ||
          !parse_positive_integer(std::string_view{argv[i]},
                                  &options->generation)) {
        *error = "--generation requires a positive 64-bit integer";
        return false;
      }
      saw_generation = true;
    } else {
      *error = "unknown argument: ";
      error->append(argument);
      return false;
    }
  }
  return true;
}

bool install_signal_handlers(std::string* error) {
  struct sigaction action {};
  action.sa_handler = request_stop;
  if (::sigemptyset(&action.sa_mask) != 0) {
    *error = std::string{"sigemptyset: "} + std::strerror(errno);
    return false;
  }
  action.sa_flags = 0;
  if (::sigaction(SIGINT, &action, nullptr) != 0) {
    *error = std::string{"sigaction(SIGINT): "} + std::strerror(errno);
    return false;
  }
  if (::sigaction(SIGTERM, &action, nullptr) != 0) {
    *error = std::string{"sigaction(SIGTERM): "} + std::strerror(errno);
    return false;
  }
  return true;
}

enum class CommandAction { continue_loop, shutdown };

CommandAction process_command(std::string command, B70Provider& provider) {
  if (!command.empty() && command.back() == '\r') {
    command.pop_back();
  }

  if (command == "capability") {
    write_capability(std::cout, provider.capability());
  } else if (command == "health") {
    write_health(std::cout, provider.health());
  } else if (command == "shutdown") {
    provider.shutdown();
    write_health(std::cout, provider.health());
    return CommandAction::shutdown;
  } else {
    write_error(std::cout, "unknown command", command);
  }
  return CommandAction::continue_loop;
}

int run_control_loop(B70Provider& provider) {
  std::array<char, 4096> read_buffer{};
  std::string pending;

  while (stop_requested == 0) {
    struct pollfd input {
      STDIN_FILENO, POLLIN, 0
    };
    const int poll_result = ::poll(&input, 1, kSignalPollIntervalMs);
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      write_error(std::cerr, "stdin poll failed", std::strerror(errno));
      provider.shutdown();
      return 1;
    }
    if ((input.revents & (POLLERR | POLLNVAL)) != 0) {
      write_error(std::cerr, "stdin became unavailable");
      provider.shutdown();
      return 1;
    }
    if ((input.revents & (POLLIN | POLLHUP)) == 0) {
      continue;
    }

    const ssize_t bytes_read =
        ::read(STDIN_FILENO, read_buffer.data(), read_buffer.size());
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      write_error(std::cerr, "stdin read failed", std::strerror(errno));
      provider.shutdown();
      return 1;
    }
    if (bytes_read == 0) {
      if (!pending.empty() &&
          process_command(std::move(pending), provider) ==
              CommandAction::shutdown) {
        return 0;
      }
      provider.shutdown();
      return 0;
    }

    pending.append(read_buffer.data(), static_cast<std::size_t>(bytes_read));
    std::size_t newline = 0;
    while ((newline = pending.find('\n')) != std::string::npos) {
      std::string command = pending.substr(0, newline);
      pending.erase(0, newline + 1);
      if (process_command(std::move(command), provider) ==
          CommandAction::shutdown) {
        return 0;
      }
      if (stop_requested != 0) {
        break;
      }
    }
  }

  provider.shutdown();
  return 0;
}

}  // namespace

int main(const int argc, char* argv[]) {
  Options options;
  std::string error;
  if (!parse_options(argc, argv, &options, &error)) {
    write_error(std::cerr, "invalid arguments", error);
    return 2;
  }
  if (!install_signal_handlers(&error)) {
    write_error(std::cerr, "signal handler setup failed", error);
    return 1;
  }

  try {
    B70Provider provider;
    ProviderConfig config;
    config.max_batch = options.max_batch;
    config.top_k = 8;
    config.generation = options.generation;

    const ProviderStatus load_status = provider.load(options.bank_path, config);
    if (load_status != ProviderStatus::ok) {
      write_error(std::cerr, "provider load failed",
                  provider.health().last_error, status_name(load_status));
      return 1;
    }

    write_capability(std::cout, provider.capability());
    return run_control_loop(provider);
  } catch (const std::exception& exception) {
    write_error(std::cerr, "provider startup failed", exception.what());
  } catch (...) {
    write_error(std::cerr, "provider startup failed",
                "unknown exception");
  }
  return 1;
}
