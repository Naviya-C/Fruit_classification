import torch
import torch.nn as nn
import torch.optim as optim
from model_train.model_setup import transfer_model
from preprocess.data_load import train_load, val_load, num_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = num_class()
model = transfer_model(num_classes)

# used to multiple gpu acceleration
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

if isinstance(model, nn.DataParallel):
    optimizer = optim.Adam(model.module.fc.parameters(), lr=0.0003) # low lr 
else:
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0003)


epochs = 4
best_acc = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_load:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clear old gradients because can accumulates gradient.

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() # Accumulate loss

        _, preds = torch.max(outputs, 1) # _ get value
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    avg_loss = total_loss / len(train_load)

    model.eval() #put

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_load:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    if val_acc > best_acc:
        best_acc = val_acc
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), "best_model.pth")
        else:
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), "final_model.pth")
else:
    torch.save(model.state_dict(), "final_model.pth")

print("Training complete. Models saved!")
