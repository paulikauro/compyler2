# compyler2
A simple compiler for a simple language. Core language features are borrowed from C, and the simple syntax is inspired by Python, Nim and Haskell. Exceptions are adopted in order to simplify error handling. The `defer` and `errdefer` keywords, from the Zig language, complement error handling, providing a nice way to deal with resource management.

# TODO
- [x] Lexer
- [ ] Parser
- [ ] Semantic analysis
- [ ] Intermediate code generation
- [ ] IR optimizations
- [ ] C backend
- [ ] x86_64 backend
- [ ] LLVM backend

**Why so many backends?** Mostly to learn.
