/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>

namespace lfs::core {

    // A tiny cross-platform helper for a single "overlay" line (e.g. a progress bar)
    // that is repeatedly redrawn in-place. While an overlay is active, any other
    // stdout/stderr messages should temporarily clear the overlay line, print the
    // message(s), and then redraw the overlay so it stays visually at the bottom.
    //
    // This intentionally avoids ANSI cursor movement so it works on Windows
    // terminals without requiring VT sequence enabling.
    class ConsoleOverlay final {
    public:
        using CallbackFn = void (*)(void*);

        static ConsoleOverlay& instance() {
            static ConsoleOverlay inst;
            return inst;
        }

        // Register callbacks for the active overlay.
        // - ctx: user pointer passed to callbacks
        // - clear: should erase the overlay line and leave cursor at line start
        // - render: should redraw the overlay at the current cursor line
        void set(void* ctx, CallbackFn clear, CallbackFn render) {
            ctx_.store(ctx, std::memory_order_release);
            clear_.store(clear, std::memory_order_release);
            render_.store(render, std::memory_order_release);
        }

        // Remove callbacks if the currently-registered ctx matches.
        void reset_if_ctx(void* ctx) {
            if (ctx_.load(std::memory_order_acquire) != ctx) {
                return;
            }
            render_.store(nullptr, std::memory_order_release);
            clear_.store(nullptr, std::memory_order_release);
            ctx_.store(nullptr, std::memory_order_release);
        }

        // Called before printing a regular line while an overlay is active.
        void before_write() {
            if (depth_++ > 0) {
                return;
            }
            const auto clear = clear_.load(std::memory_order_acquire);
            if (!clear) {
                return;
            }
            void* ctx = ctx_.load(std::memory_order_acquire);
            if (!ctx) {
                return;
            }
            clear(ctx);
        }

        // Called after printing a regular line while an overlay is active.
        void after_write() {
            if (--depth_ > 0) {
                return;
            }
            const auto render = render_.load(std::memory_order_acquire);
            if (!render) {
                return;
            }
            void* ctx = ctx_.load(std::memory_order_acquire);
            if (!ctx) {
                return;
            }
            render(ctx);
        }

        // RAII helper for non-logger code paths doing raw stdio writes.
        class ScopedSuspend final {
        public:
            ScopedSuspend() { ConsoleOverlay::instance().before_write(); }
            ~ScopedSuspend() { ConsoleOverlay::instance().after_write(); }
            ScopedSuspend(const ScopedSuspend&) = delete;
            ScopedSuspend& operator=(const ScopedSuspend&) = delete;
        };

    private:
        ConsoleOverlay() = default;

        std::atomic<void*> ctx_{nullptr};
        std::atomic<CallbackFn> clear_{nullptr};
        std::atomic<CallbackFn> render_{nullptr};

        inline static thread_local int depth_ = 0;
    };

} // namespace lfs::core
