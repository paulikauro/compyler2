# compyler2
A simple compiler for a simple language. Core language features are borrowed from C, and the simple syntax is inspired by Python, Nim and Haskell. Exceptions are adopted in order to simplify error handling. The `defer` and `errdefer` keywords, from the Zig language, complement error handling, providing a nice way to deal with resource management.

# TODO
- [x] Lexer
- [x] Parser
    - mostly finished
- [ ] Semantic analysis
- [ ] Intermediate code generation
- [ ] IR optimizations
- [ ] C backend
    - intermediate goal to get something concrete done sooner (may be removed)
- [ ] x86_64 backend
    - one of the final goals of this project
- [ ] Possibly other backends
- [ ] LLVM backend
    - an extra goal

# Planned language features
These do not have a high priority, the language already has more features than necessary for this project.

- multi-dimensional arrays (may be added soon)
- generics
- function overloading & possibly some way of neatly dispatching them based on a tag field in a struct
