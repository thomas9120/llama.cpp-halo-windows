// Compile with -fno-access-control to exercise the server's private batch recovery without loading a model.
#include "tools/server/server-context.cpp"

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

template <typename Slots>
static void check_iteration(server_context_impl & server, Slots & slots) {
    server.batch.clear();
    int visited = 0;
    bool propagated = false;
    try {
        server.iterate(slots, [&](server_slot & slot) {
            ++visited;
            server.batch.add(slot.id, 1, 0, false, true);
            throw std::bad_alloc();
        });
    } catch (const std::bad_alloc &) {
        propagated = true;
    }
    check(propagated, "allocation failure was swallowed during batch construction");
    check(visited == 1, "iteration continued after a partial batch failure");
    check(server.batch.size() == 1 && !server.batch.slot_batched, "fault did not reproduce a partial batch");

    server.batch.clear();
    server.iterate(slots, [&](server_slot & slot) {
        server.batch.add(slot.id, 1, 0, false, true);
        server.batch.slot_batched = &slot;
    });
    server.batch.render();
    check(server.batch.size() == 2 && server.batch.slot_batched, "next batch did not recover");
}

int main() {
    try {
        server_context_impl server;
        server.batch.init(4, 0);
        server.slots.resize(2);
        server.slots[0].id = 0;
        server.slots[1].id = 1;
        for (auto & slot : server.slots) {
            slot.task = std::make_unique<server_task>(SERVER_TASK_TYPE_COMPLETION);
        }
        check_iteration(server, server.slots);
        std::vector<server_slot *> pointers { &server.slots[0], &server.slots[1] };
        check_iteration(server, pointers);
        printf("PASS: both slot iterators propagate allocation failures and allow a fresh batch\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
