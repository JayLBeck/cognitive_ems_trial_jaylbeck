#!/usr/bin/env python3
"""
Standalone raw SRT client for testing EgoEMS-Sim without EMS-Pipeline.

It connects to the simulator server, reassembles the custom video/audio/CSV
frames, validates payload sizes, and prints receive statistics.
"""
import argparse
import ctypes
import ctypes.util
import socket
import struct
import sys
import time

import numpy as np
import cv2


def load_libsrt():
    candidates = [
        ctypes.util.find_library("srt"),
        ctypes.util.find_library("srt-gnutls"),
        ctypes.util.find_library("srt-openssl"),
        "libsrt.so.1",
        "libsrt.so",
        "libsrt.dylib",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(
            "libsrt not found. Install libsrt or set your library path so ctypes can load it."
        )

    lib.srt_startup.argtypes = []
    lib.srt_startup.restype = ctypes.c_int
    lib.srt_create_socket.argtypes = []
    lib.srt_create_socket.restype = ctypes.c_int
    lib.srt_setsockopt.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.srt_setsockopt.restype = ctypes.c_int
    lib.srt_connect.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.srt_connect.restype = ctypes.c_int
    lib.srt_recv.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.srt_recv.restype = ctypes.c_int
    lib.srt_close.argtypes = [ctypes.c_int]
    lib.srt_close.restype = ctypes.c_int
    lib.srt_cleanup.argtypes = []
    lib.srt_cleanup.restype = ctypes.c_int
    return lib


if sys.platform == "darwin":
    class SockaddrIn(ctypes.Structure):
        _fields_ = [
            ("sin_len", ctypes.c_uint8),
            ("sin_family", ctypes.c_uint8),
            ("sin_port", ctypes.c_uint16),
            ("sin_addr", ctypes.c_byte * 4),
            ("sin_zero", ctypes.c_byte * 8),
        ]
else:
    class SockaddrIn(ctypes.Structure):
        _fields_ = [
            ("sin_family", ctypes.c_short),
            ("sin_port", ctypes.c_ushort),
            ("sin_addr", ctypes.c_byte * 4),
            ("sin_zero", ctypes.c_byte * 8),
        ]


def build_addr(host, port):
    ip = socket.gethostbyname(host)
    parts = [int(x) for x in ip.split(".")]
    addr = SockaddrIn()
    if sys.platform == "darwin":
        addr.sin_len = ctypes.sizeof(SockaddrIn)
        addr.sin_family = socket.AF_INET
    else:
        addr.sin_family = socket.AF_INET
    addr.sin_port = socket.htons(port)
    addr.sin_addr = (ctypes.c_byte * 4)(*parts)
    return addr, ip


def recv_chunk(libsrt, sock):
    buf = ctypes.create_string_buffer(2048)
    received = libsrt.srt_recv(sock, buf, 2048)
    if received <= 0:
        return None
    if received < 11:
        return None

    data_type, frame_idx, chunk_num, total_chunks, data_size = struct.unpack(
        "!BIHHH", buf.raw[:11]
    )
    return data_type, frame_idx, chunk_num, total_chunks, buf.raw[11:11 + data_size]


def connect(libsrt, host, port):
    if libsrt.srt_startup() < 0:
        raise RuntimeError("SRT startup failed")

    sock = libsrt.srt_create_socket()
    if sock < 0:
        libsrt.srt_cleanup()
        raise RuntimeError("SRT socket creation failed")

    rcvbuf = ctypes.c_int(96_000_000)
    libsrt.srt_setsockopt(sock, 0, 8, ctypes.byref(rcvbuf), ctypes.sizeof(rcvbuf))

    rcvlatency = ctypes.c_int(500)
    libsrt.srt_setsockopt(sock, 0, 43, ctypes.byref(rcvlatency), ctypes.sizeof(rcvlatency))

    addr, ip = build_addr(host, port)
    print(f"[client] Connecting to {ip}:{port}...")
    if libsrt.srt_connect(sock, ctypes.byref(addr), ctypes.sizeof(addr)) < 0:
        libsrt.srt_close(sock)
        libsrt.srt_cleanup()
        raise RuntimeError("SRT connect failed")

    print("[client] Connected")
    return sock


def run(args):
    libsrt = load_libsrt()
    sock = connect(libsrt, args.host, args.port)

    expected_video_size = args.width * args.height * 3
    needed_types = {1, 2, 3}
    frame_buffers = {}
    expected_chunks = {}
    current_frame = None
    frames = 0
    bad_frames = 0
    start = time.time()
    last_report = start

    try:
        while args.frames <= 0 or frames < args.frames:
            chunk = recv_chunk(libsrt, sock)
            if chunk is None:
                print("[client] Connection closed")
                break

            data_type, frame_idx, chunk_num, total_chunks, data = chunk
            if data_type not in needed_types:
                continue

            frame_buffers.setdefault(frame_idx, {1: {}, 2: {}, 3: {}})
            expected_chunks.setdefault(frame_idx, {})
            if current_frame is None:
                current_frame = frame_idx

            frame_buffers[frame_idx][data_type][chunk_num] = data
            expected_chunks[frame_idx][data_type] = total_chunks

            if current_frame not in frame_buffers:
                continue

            fb = frame_buffers[current_frame]
            ec = expected_chunks[current_frame]
            complete = all(
                data_type in ec and len(fb[data_type]) == ec[data_type]
                for data_type in needed_types
            )
            if not complete:
                continue

            payloads = {
                data_type: b"".join(fb[data_type][i] for i in range(ec[data_type]))
                for data_type in needed_types
            }

            video_ok = len(payloads[1]) == expected_video_size
            csv_text = payloads[3].decode("utf-8", errors="replace").strip()
            if not video_ok:
                bad_frames += 1
                print(
                    f"[client] Bad video size frame={current_frame}: "
                    f"{len(payloads[1])} != {expected_video_size}"
                )
            elif frames == 0:
                print(
                    f"[client] First frame OK: video={len(payloads[1])} bytes, "
                    f"audio={len(payloads[2])} bytes, csv={csv_text}"
                )

#-----------Part2 Modifications-------------------------------------------------------------------------------
            elif frames % 5 == 0:
                image = np.frombuffer(payloads[1], dtype=np.uint8)
                image = image.reshape((args.height, args.width, 3))

                cv2.imshow("frame", image)
                cv2.waitKey(1)

#-----------End of Modifications------------------------------------------------------------------------------

            frames += 1
            now = time.time()
            if now - last_report >= 1.0:
                elapsed = now - start
                fps = frames / elapsed if elapsed > 0 else 0.0
                print(
                    f"[client] frames={frames} fps={fps:.1f} "
                    f"bad={bad_frames} last_csv={csv_text}"
                )
                last_report = now

            del frame_buffers[current_frame]
            del expected_chunks[current_frame]
            current_frame += 1

            stale = [idx for idx in frame_buffers if idx < current_frame - 5]
            for idx in stale:
                del frame_buffers[idx]
                expected_chunks.pop(idx, None)

    except KeyboardInterrupt:
        print("\n[client] Stopped by user")
    finally:
        libsrt.srt_close(sock)
        libsrt.srt_cleanup()
        elapsed = max(time.time() - start, 1e-6)
        print(f"[client] Done: frames={frames} fps={frames / elapsed:.1f} bad={bad_frames}")


def main():
    parser = argparse.ArgumentParser(description="Test the EgoEMS-Sim raw SRT stream.")
    parser.add_argument("host", help="Simulator host/IP")
    parser.add_argument("--port", type=int, default=9000, help="Simulator SRT port")
    parser.add_argument("--width", type=int, default=480, help="Expected video width")
    parser.add_argument("--height", type=int, default=270, help="Expected video height")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means run until stream ends",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
