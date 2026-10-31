import SwiftUI

@main
struct SubvoxApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .frame(minWidth: 980, minHeight: 680)
                .task { model.start() }
        }
        .windowResizability(.contentMinSize)

        Window("Avatar", id: "avatar") {
            AvatarWindow()
                .environment(model)
        }
        .defaultSize(width: 480, height: 360)
    }
}
