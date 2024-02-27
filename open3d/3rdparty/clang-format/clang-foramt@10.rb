# BSD 2-Clause License
#
# Copyright (c) 2009-present, Homebrew contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Ref:
# https://github.com/Homebrew/homebrew-core/blob/7f9aa30778d3e3c391b0060a1ac4af230f351dcc/Formula/clang-format.rb#L1

class ClangFormatAT10 < Formula
    desc "Formatting tools for C, C++, Obj-C, Java, JavaScript, TypeScript"
    homepage "https://clang.llvm.org/docs/ClangFormat.html"
    # The LLVM Project is under the Apache License v2.0 with LLVM Exceptions
    license "Apache-2.0"
    version "10.0.1"
    head "https://github.com/llvm/llvm-project.git"

    stable do
      url "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/llvm-10.0.1.src.tar.xz"
      sha256 "c5d8e30b57cbded7128d78e5e8dad811bff97a8d471896812f57fa99ee82cdf3"

      resource "clang" do
        url "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/clang-10.0.1.src.tar.xz"
        sha256 "f99afc382b88e622c689b6d96cadfa6241ef55dca90e87fc170352e12ddb2b24"
      end

      resource "libcxx" do
        url "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/libcxx-10.0.1.src.tar.xz"
        sha256 "def674535f22f83131353b3c382ccebfef4ba6a35c488bdb76f10b68b25be86c"
      end
    end

    bottle do
      cellar :any_skip_relocation
      sha256 "65667bcca091df96d9f27b56a7726a41c9998ebbcd3b10bc2eb2d43aa871c216" => :catalina
      sha256 "9b711f49db65634cb1dca8149804040f337ce3e68e66eeba7270c2aec66b90e0" => :mojave
      sha256 "9bbc58f9d5afb228c2aeda8ea03d305d7912860f27941933d0824bee505752bd" => :high_sierra
    end

    depends_on "cmake" => :build
    depends_on "ninja" => :build

    uses_from_macos "libxml2"
    uses_from_macos "ncurses"
    uses_from_macos "zlib"

    def install
      if build.head?
        ln_s buildpath/"libcxx", buildpath/"llvm/projects/libcxx"
        ln_s buildpath/"clang", buildpath/"llvm/tools/clang"
      else
        (buildpath/"projects/libcxx").install resource("libcxx")
        (buildpath/"tools/clang").install resource("clang")
      end

      llvmpath = build.head? ? buildpath/"llvm" : buildpath

      mkdir llvmpath/"build" do
        args = std_cmake_args
        args << "-DLLVM_ENABLE_LIBCXX=ON"
        args << ".."
        system "cmake", "-G", "Ninja", *args
        system "ninja", "clang-format"
      end

      bin.install llvmpath/"build/bin/clang-format"
      bin.install llvmpath/"tools/clang/tools/clang-format/git-clang-format"
      (share/"clang").install Dir[llvmpath/"tools/clang/tools/clang-format/clang-format*"]
    end

    test do
      # NB: below C code is messily formatted on purpose.
      (testpath/"test.c").write <<~EOS
        int         main(char *args) { \n   \t printf("hello"); }
      EOS

      assert_equal "int main(char *args) { printf(\"hello\"); }\n",
          shell_output("#{bin}/clang-format -style=Google test.c")
    end
  end
